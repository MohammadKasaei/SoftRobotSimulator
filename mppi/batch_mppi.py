import torch
from dataclasses import dataclass

@dataclass
class MPPIParams:
    T: int = 0
    K: int = 0
    batch: int = 0
    temp: float = 0
    sigma: float = 0
    delta: float = 0.01
    discount: float = 1.01


class MPPIController:
    def __init__(self, dynamics, r_cost, t_cost, u_cost, params: MPPIParams, dims, device='cuda:0'):
        self._nu, self._nx = dims
        self._dynamics = dynamics
        self._r_cost = r_cost
        self._t_cost = t_cost
        self._u_cost = u_cost
        self._params = params
        self._states = torch.empty((self._params.K, self._params.T+1, self._nx)).to(device)
        self._perts = torch.empty(((self._params.K, self._params.T, self._nu))).to(device)
        self._new_u = torch.empty(((self._params.K, self._params.T, self._nu))).to(device)
        self._ctrls = torch.zeros(((self._params.T, self._nu))).to(device)
        self._costs = torch.zeros(((self._params.K))).to(device)
        self._norm = None

    def sample_ctrl(self, u):
        perts = torch.randn_like(self._perts) * self._params.sigma

        return perts

    def rollout(self, x, u):
        disc = 1
        x_init = x.reshape(self._params.K, self._nx)
        self._perts = self.sample_ctrl(u)
        self._states[:, 0] = x_init
        self._costs += self._r_cost(x_init).squeeze()
        for t in range(self._params.T):
            temp_ctrl = u[t]
            temp_pert = self._perts[:, t].reshape(self._params.K, self._nu) 
            self._new_u[:, t] = (temp_pert + temp_ctrl).reshape(self._new_u[:, t].shape)
            
            xd = self._dynamics(x_init, self._new_u[:, t]).reshape(self._params.K, self._nx)
            x_init = x_init + xd * self._params.delta
            self._states[:, t+1] = x_init
            u_reg = self._params.temp * self._u_cost(self._new_u[:, t], self._perts[:, t])
            self._costs += self._r_cost(x_init).squeeze() * disc + u_reg.squeeze()
            disc *= self._params.discount

        self._costs += self._t_cost(self._states[:, -1]).squeeze() * (disc * self._params.discount)

    def compute_ctrls(self, u):
        min_cst = self._costs.min()
        batch_norm = torch.sum(torch.exp(-1 / self._params.temp * (self._costs - min_cst)), dim=0).squeeze()
        weights = torch.exp(-1 / self._params.temp * (self._costs - min_cst))/batch_norm.unsqueeze(-1)
        self._norm = weights
        weights = weights.unsqueeze(1)
        for t in range(self._params.T):
            batch_ctrl = torch.sum(weights * self._perts[:, t], dim=0)
            u[t] += batch_ctrl
        
        self._costs *= 0
        return u

    def MPC(self, x):
        u = self._ctrls
        xb = x.clone().repeat(self._params.K, 1, 1)
        self.rollout(xb, u)
        self._ctrls = self.compute_ctrls(u)        
        u = self._ctrls[0, :]
        # self._ctrls = self._ctrls.roll(-1, dims=0)
        # self._ctrls[-1, :] *= 0
        return u
