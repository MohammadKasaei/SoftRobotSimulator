import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim



import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchdiffeq import odeint, odeint_adjoint
from   scipy.integrate import solve_ivp

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
device =  'cpu'


t = torch.linspace(0., 0.1, 2).to(device)
k = np.array([40., 25., 40., 5.]) 


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 4), facecolor='white')
ax_traj = fig.add_subplot(111, frameon=False)
plt.show(block=False)


# Simulate the dynamics
class Lambda(nn.Module):
    def __init__(self):
        super(Lambda, self).__init__()
        self.l0 = 100e-3
        # cables offset
        self.d  = 7.5e-3
        # ode step time
        self.ds = 0.005
        
        r0 = np.array([0,0,0]).reshape(3,1)  
        R0 = np.eye(3,3)
        R0 = np.reshape(R0,(9,1))
        y0 = np.concatenate((r0, R0), axis=0)
        self.y0 = np.squeeze(np.asarray(y0))
        

    def odeFunction(self,s,y):
        dydt  = np.zeros(12)
        y = y.numpy()
        # % 12 elements are r (3) and R (9), respectively
        e3    = np.array([0,0,1]).reshape(3,1)              
        u_hat = np.array([[0,0,self.uy], [0, 0, -self.ux],[-self.uy, self.ux, 0]])
        r     = y[0:3].reshape(3,1)
        R     = np.array( [y[3:6],y[6:9],y[9:12]]).reshape(3,3)
        # % odes
        dR  = R @ u_hat
        dr  = R @ e3
        dRR = dR.T
        dydt[0:3]  = dr.T
        dydt[3:6]  = dRR[:,0]
        dydt[6:9]  = dRR[:,1]
        dydt[9:12] = dRR[:,2]
        return dydt


    def forward(self, t, y):
        action = np.array((t,y[0],y[1]))
        self.l  = self.l0 
        self.uy = ( action[1]) /  (self.l * self.d)
        self.ux = ( action[2]) / -(self.l * self.d)
        dyds = self.odeFunction(t,y[2:])
        du = np.array((0,0))
        du_y = torch.from_numpy(np.concatenate((du,dyds)))

        return du_y


# Here we used this function to sample from the model instead of the real data as our dataset is for our robot 
def sample_from_a_model(seed = 0):
    np.random.seed(seed) # fixed the seed for reproducibility
    
    with torch.no_grad():
        ux = np.random.uniform(-0.015,0.015)
        uy = np.random.uniform(-0.015,0.015)

        r0 = np.array([0,0,0]).reshape(3,1)  
        R0 = np.eye(3,3)
        R0 = np.reshape(R0,(9,1))
        y0 = np.concatenate((r0, R0), axis=0)
        y0 = np.squeeze(np.asarray(y0))
        u0 = np.array((ux,uy))
        u0y0 = np.concatenate((u0,y0))

        x0 = torch.tensor(u0y0, dtype=torch.float32).to(device)
        true_y = odeint(Lambda(), x0, t,method = 'fixed_adams')
        
  
    return x0[0:5].to(device), t.to(device), true_y[:,0:5].to(device), x0[0:5], true_y[:,0:5]




# Neural ODE Network
class ODEFunc(nn.Module):

    "solve the problem of dimension"

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.lin1 = nn.Linear(5, 64)
        self.relu1 =  nn.ELU()
        self.lin2 = nn.Linear(64, 48)
        self.relu2 = nn.ELU()
        self.lin3 = nn.Linear(48, 16)
        self.relu3 = nn.ELU()
        self.lin4 = nn.Linear(16, 5)

     

    def forward(self, t, y):
        ####original###
        y = torch.squeeze(y)
        y = self.relu1(self.lin1(y))
        y = self.relu2(self.lin2(y))
        y = self.relu3(self.lin3(y))
        y = self.lin4(y)
        return y


#Not Important
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def visualize(true_y, pred_y, odefunc, itr):
    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('length (m)')
    ax_traj.set_ylabel('position (m)')
    
    ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 2], 'r-', label = 'predict_x')
    ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 3], 'g-', label = 'predict_y')
    ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 4], 'b-', label = 'predict_z')
    
    ax_traj.plot(t.cpu().numpy(),true_y.cpu().numpy()[:, 2], 'r--', label='true_x')
    ax_traj.plot(t.cpu().numpy(),true_y.cpu().numpy()[:, 3], 'g--', label='true_y')
    ax_traj.plot(t.cpu().numpy(),true_y.cpu().numpy()[:, 4], 'b--', label='true_z')

    ax_traj.legend()
    fig.tight_layout()
    plt.draw()
    plt.pause(0.1)




if __name__ == '__main__':

    
    func = ODEFunc().to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    Loss_creterion = nn.MSELoss()
    idx = 0
 
    for itr in range(1, 50000):
        optimizer.zero_grad()
        
        batch_y0, batch_t, batch_y, x_start, true_y= sample_from_a_model(seed = itr%25) # fixed the seed to get the same data for each iteration
        pred_y = odeint(func, batch_y0, batch_t)
        loss = Loss_creterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % 10 == 0:
            with torch.no_grad():
                pred_y = odeint(func, x_start, t)
                loss = Loss_creterion(pred_y, true_y)
                print('Iter {:04d} | Test Loss:{:.6f} |Total Loss {:.6f}'.format(itr, loss.item(),loss_meter.val))
                visualize(true_y, pred_y, func, idx)
                idx += 1

        end = time.time()

        if itr % 1000 == 0:
            timestr   = time.strftime("%Y%m%d-%H%M%S")
            modelName = "neuralODE/trainedModels/model_NODE_REDU_"+ timestr+f"_T_{itr}.zip"
            torch.save(func.state_dict(), modelName)
            print (f"temporary model: {modelName} has been saved.")
    
    
    timestr   = time.strftime("%Y%m%d-%H%M%S")
    modelName = "neuralODE/trainedModels/model_NODE_REDU_"+ timestr+".zip"
    torch.save(func.state_dict(), modelName)
    print (f"model: {modelName} has been saved.")

    plt.show()
