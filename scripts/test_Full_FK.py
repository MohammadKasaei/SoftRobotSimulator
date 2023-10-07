import cProfile



from visualizer.visualizer import ODE
from visualizer.visualizer import softRobotVisualizer
import time
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pytorch_mppi import mppi

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

# Neural ODE Network
class ODEFunc(nn.Module):

    "solve the problem of dimension"

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.lin1 = nn.Linear(5, 64)
        self.relu1 = nn.ELU()
        self.lin2 = nn.Linear(64, 48)
        self.relu2 = nn.ELU()
        self.lin3 = nn.Linear(48, 16)
        self.relu3 = nn.ELU()
        self.lin4 = nn.Linear(16, 5)

    def forward(self, t, y):
        # ####original###
        y = torch.squeeze(y.float())
        y = self.relu1(self.lin1(y))
        y = self.relu2(self.lin2(y))
        y = self.relu3(self.lin3(y))
        y = self.lin4(y)

        return y




class MPPISystem():
    def __init__(self) -> None:
        
        # self.timeSteps = torch.linspace(0.0, self.ts, 2).to(device)
        self.nx = 3
        self.nu = 3
        self.Jac = None
        self.ts = 0.05
        # self.xc     = torch.tensor((0,0,0.1),device=device) 
        self.ref    = torch.tensor([0,0.0,0.1],device=device)
        self.lastU  = torch.tensor([0,0,0],device=device)
        self.obs1   = 0*torch.tensor([0.03,0,0.1],device=device)
        self.obs2   = 0*torch.tensor([-0.03,0,0.11],device=device)
        self.obs3   = 0*torch.tensor([0.03,0,0.1],device=device)
        self.obs4   = 0*torch.tensor([-0.03,0,0.11],device=device)
        self._idx   = 0

    def updateObsPos(self,id,pos):
        if id == 1:
            self.obs1   = torch.tensor(pos,device=device)
        elif id == 2:
            self.obs2   = torch.tensor(pos,device=device)
        elif id == 3:
            self.obs3   = torch.tensor(pos,device=device)
        elif id == 4:
            self.obs4   = torch.tensor(pos,device=device)

    def Dynamcis(self,x,u):
        u[:,0] = torch.clamp(u[:,0], -0.03, 0.03)
        u[:,1] = torch.clamp(u[:,1], -0.015, 0.015)
        u[:,2] = torch.clamp(u[:,2], -0.015, 0.015)
        
        xdot = (self.Jac.mm(u.T)).T
        self.x = x + xdot*self.ts

        return self.x

    def running_cost(self,states, actions):
        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]
        d1 = 0*torch.linalg.norm (states-self.obs1,dim=1)
        d2 = 0*torch.linalg.norm (states-self.obs2,dim=1)
        cost = 200 * (torch.abs(x-self.ref[0,self._idx])) + 200 *torch.abs(y-self.ref[1,self._idx]) + 200*torch.abs(z-self.ref[2,self._idx]) + \
                100000 * torch.lt(d1,0.01) + 100000 * torch.lt(d2,0.01)     

        self._idx +=1
                
        # cost = 1000*(torch.abs(x-self.ref[0])) + 1000*torch.abs(y-self.ref[1]) + 1000*torch.abs(z-self.ref[2]) 
        return cost
    
    def terminal_state_cost (self,states, actions):
        x = states[0,:,-1, 0]
        y = states[0,:,-1, 1]
        z = states[0,:,-1, 2]
        # action = action[:, 0]
        cost = 100*torch.abs(x-self.ref[0,-1]) + 100*torch.abs(y-self.ref[1,-1]) + 100*torch.abs(z-self.ref[2,-1]) 
        self._idx = 0
        return cost

class prediction():
    def __init__(self, modelPath) -> None:
        self.l0 = 0.1
        # self.ds = 0.001*20
        self.ds = 0.01
        
        q = torch.tensor((0, 0, 0))
        self.updateX0(q)

        self.func = ODEFunc().to(device)
        self.func.load_state_dict(torch.load(modelPath))
        self._mlp_model  = torch.load("neuralODE/trainedModels/model_FK_FullMLP.zip")
        self._mlp_model.eval()

    def updateX0(self, q):
        r0 = torch.tensor([0, 0, 0],dtype=torch.float).reshape(3, 1)
        R0 = torch.eye(3, 3)
        R0 = torch.reshape(R0, (9, 1))
        y0 = torch.cat((r0, R0), axis=0)
        y0 = torch.squeeze(y0)
        y0 = torch.tensor((0,0,0))
        u0 = torch.tensor((q[1], q[2]))
        
        u0y0 = torch.cat((u0, y0))
        self.x0 = u0y0.to(device)
        

    def predictMLP(self, q):
        return self._mlp_model(q.float())
    
    def MLP_Jac(self, q, dq=torch.tensor((1e-4, 1e-4, 1e-4),dtype=torch.float)):
        #fx0 = self.predict(q,True)
        n = len(q)
        m = 3  # len(fx0)
        jac = torch.zeros((n, m)).to(device)
        for j in range(m):  # through rows
            if (j == 0):
                Dq = torch.tensor((dq[0]/2.0, 0, 0))
            elif (j == 1):
                Dq = torch.tensor((0, dq[1]/2.0, 0))
            elif (j == 2):
                Dq = torch.tensor((0, 0, dq[2]/2.0))

            jac[j, :] = (self.predictMLP(torch.tensor(q)+Dq) -
                         self.predictMLP(torch.tensor(q)-Dq))/dq[j]
        return jac

        
    
    def predict(self, q, returnJustEE=False):
        self.updateX0(q)
        l = self.l0+q[0]

        timeSteps = torch.linspace(0.0, l, int(l/self.ds)).to(device)
        # prediction = odeint_adjoint(self.func, self.x0, timeSteps,method='explicit_adams').cpu().numpy()
        
        # prediction = odeint(self.func, self.x0, timeSteps,method='explicit_adams')
        prediction = odeint(self.func, self.x0, timeSteps,method='euler')
        

        return prediction[-1, 2:] if returnJustEE else prediction
    
    def Jac(self, q, dq=torch.tensor((1e-4, 1e-4, 1e-4),dtype=torch.float)):
        #fx0 = self.predict(q,True)
        n = len(q)
        m = 3  # len(fx0)
        jac = torch.zeros((n, m)).to(device)
        for j in range(m):  # through rows
            if (j == 0):
                Dq = torch.tensor((dq[0]/2.0, 0, 0))
            elif (j == 1):
                Dq = torch.tensor((0, dq[1]/2.0, 0))
            elif (j == 2):
                Dq = torch.tensor((0, 0, dq[2]/2.0))

            jac[j, :] = (self.predict(torch.tensor(q)+Dq, True) -
                         self.predict(torch.tensor(q)-Dq, True))/dq[j]
        return jac



def get_ref(gt,horizon = 10, traj = 'rose'):
        
        if traj == 'rose':
            # ################################### rose ##########################################
            # r = a * cos(k * theta)
            # x = r * cos(theta)
            # y = r * sin(theta)
            # Where:
            # a: the amplitude of the curve
            # k: the frequency of the petals
            # theta: the angle in radians

            k = 4
            T  = 100
            w  = 2*np.pi/T
            a = 0.025
            r  = a * np.cos(k*w*gt)
            xd = (x0 + np.array((r*np.cos(w*gt),r*np.sin(w*gt),0.00*gt)))
            xd_dot = np.array((-r*w*np.sin(w*gt),r*w*np.cos(w*gt),0.00*gt))
            #mppi
            gtt = np.linspace(gt,gt+(horizon*ts),horizon) 
            xx0 = x0.reshape(3,1)
            ref =  torch.tensor(xx0  + np.array((r*np.cos(w*gtt),r*np.sin(w*gtt),0.00*gtt)),device=device)            

        elif traj == 'limacon':
            # ################################### limaçon ##########################################
            # A limaçon trajectory defined by the equations:
            # x(t) = (b+acos(t))cos(t) , y(t)= (b+acos(t))sin(t)
            # where 'a' and 'b' are constants determining the shape of the limaçon and 't' is the parameter.

            T  = 100
            w  = 2*np.pi/T
            radius = 0.02
            radius2 = 0.03
            shift = -0.02
            xd = (x0 + np.array(((shift+(radius+radius2*np.cos(w*gt))*np.cos(w*gt)),(radius+radius2*np.cos(w*gt))*np.sin(w*gt),0.00*gt)))
            xd_dot = np.array((radius*(-w*np.sin(w*(gt)-0.5*w*np.sin(w/2*(gt)))),radius*(w*np.cos(w*(gt)-0.5*radius2*np.cos(w/2*gt))),0.00))
            #mppi
            gtt = np.linspace(gt,gt+(horizon*ts),horizon) 
            xx0 = x0.reshape(3,1)
            ref =  torch.tensor(xx0  + np.array(((shift+(radius+radius2*np.cos(w*gtt))*np.cos(w*gtt)),(radius+radius2*np.cos(w*gtt))*np.sin(w*gtt),0.00*gtt)),device=device)

        elif traj == 'helix':
            ###############################  Helix #####################################
            T  = 25
            w  = 2*np.pi/T
            radius = 0.03
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),0.0005*gt)))
            xd_dot = ( np.array((radius*w*np.cos(w*(gt)),-radius*w*np.sin(w*(gt)),0.0005)))

            #mppi
            gtt = np.linspace(gt,gt+(horizon*ts),horizon) 
            xx0 = x0.reshape(3,1)
            ref =  torch.tensor(xx0 + np.array((radius*np.sin(w*(gtt)),radius*np.cos(w*(gtt)),0.0005*gtt)),device=device)
        
        ################################  Circle #####################################
        # T  = 50*2
        # w  = 2*np.pi/T
        # radius = 0.02
        # xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),0.00*gt)))
        # xd_dot = np.array((radius*w*np.cos(w*(gt)),-radius*w*np.sin(w*(gt)),0.00))

        

        ################################  Eight Figure #####################################
        # T  = 25*2
        # A  = 0.03
        # w  = 2*np.pi/T
        # xd = np.array((A*np.sin(w*gt) , A*np.sin((w/2)*gt),0.1))
        # xd_dot = np.array((A*w*np.cos(w*gt),A*w/2*np.cos(w/2*gt),0.00))
        ################################  Square #####################################
        # T  = 12.5*2
        # tt = (gt)
        # scale = 3

        # if (tt<T):
        #     xd = (x0 + scale*np.array((-0.01+(0.02/T)*tt,0.01,0.0)))
        #     xd_dot = scale*np.array(((0.02/T),0,0))
        # elif (tt<2*T):
        #     xd = (x0 + scale*np.array((0.01,0.01-((0.02/T)*(tt-T)),0.0)))
        #     xd_dot = scale*np.array((0,-(0.02/T),0))
        # elif (tt<3*T):
        #     xd = (x0 + scale*np.array((0.01-((0.02/T)*(tt-(2*T))),-0.01,0.0)))
        #     xd_dot = scale*np.array((-(0.02/T),0,0))
        # elif (tt<4*T):
        #     xd = (x0 + scale*np.array((-0.01,-0.01+((0.02/T)*(tt-(3*T))),0.0)))
        #     xd_dot = scale*np.array((0,+(0.02/T),0))
        # else:
        #     # t0 = time.time()+5
        #     gt = 0
        ################################ Moveing Square #################################
        # T  = 10.0
        # tt = (gt % (4*T))
        # if (tt<T):
        #     xd = (x0 + 2*np.array((-0.01+(0.02/T)*tt,0.01,-0.02+0.0005*gt)))
        #     xd_dot = 2*np.array(((0.02/T),0,0.0005))
        # elif (tt<2*T):
        #     xd = (x0 + 2*np.array((0.01,0.01-((0.02/T)*(tt-T)),-0.02+0.0005*gt)))
        #     xd_dot = 2*np.array((0,-(0.02/T),0.0005))
        # elif (tt<3*T):
        #     xd = (x0 + 2*np.array((0.01-((0.02/T)*(tt-(2*T))),-0.01,-0.02+0.0005*gt)))
        #     xd_dot = 2*np.array((-(0.02/T),0,0.0005))
        # elif (tt<4*T):
        #     xd = (x0 + 2*np.array((-0.01,-0.01+((0.02/T)*(tt-(3*T))),-0.02+0.0005*gt)))
        #     xd_dot = 2*np.array((0,+(0.02/T),0.0005))

        ################################ Square #################################
        # T = 10.0
        # tt = (gt % (4*T))
        # if (tt < T):
        #     xd = (x0 + 1*np.array((-0.01+(0.02/T)*tt, 0.01, 0.000*gt)))
        #     xd_dot = 1*np.array(((0.02/T), 0, 0.000))
        # elif (tt < 2*T):
        #     xd = (x0 + 1*np.array((0.01, 0.01-((0.02/T)*(tt-T)), 0.000*gt)))
        #     xd_dot = 1*np.array((0, -(0.02/T), 0.000))
        # elif (tt < 3*T):
        #     xd = (x0 + 1*np.array((0.01-((0.02/T)*(tt-(2*T))), -0.01, 0.000*gt)))
        #     xd_dot = 1*np.array((-(0.02/T), 0, 0.000))
        # elif (tt < 4*T):
        #     xd = (x0 + 1*np.array((-0.01, -0.01+((0.02/T)*(tt-(3*T))), 0.000*gt)))
        #     xd_dot = 1*np.array((0, +(0.02/T), 0.000))

        ################################  Triangle #####################################
        # T  = 12.5 *2
        # tt = (gt)
        # scale = 2

        # if (tt<T):
        #     xd = (x0 + scale*np.array((-0.01+(0.02/T)*tt,-0.01+(0.02/T)*tt,0.0)))
        #     xd_dot = scale*np.array(((0.02/T),(0.02/T),0))
        # elif (tt<2*T):
        #     xd = (x0 + scale*np.array((0.01+(0.02/T)*(tt-(T)),0.01-((0.02/T)*(tt-(T))),0.0)))
        #     xd_dot = scale*np.array(((0.02/T),-(0.02/T),0))
        # elif (tt<4*T):
        #     xd = (x0 + scale*np.array((0.03-((0.02/T)*(tt-(2*T))),-0.01,0.0)))
        #     xd_dot = scale*np.array((-(0.02/T),0,0))
        # else:
        #     # t0 = time.time()+5
        #     gt = 0

        ######################################################################################
        ######################################################################################
        ######################################################################################
        return xd,xd_dot, ref



if __name__ == '__main__':

    # Create a cProfile object
    profiler = cProfile.Profile()
    
    saveGif  = False
    animEn   = True
    saveLog  = False
    addNoise = False
    filteringObs = False
    filteringAct = False
    ## mppi
    sys = MPPISystem()
    sys.ref = torch.tensor([0.0, 0.0, 0.1],device=device)

    dtype = torch.float
    noise_mu = torch.tensor(([0, 0, 0]),dtype=dtype,device=device)
    noise_sigma = 0.0000001*torch.eye(3,dtype=dtype,device=device)
    umin = torch.tensor(([-0.01,-0.01,-0.01]),dtype=dtype,device=device)
    umax = torch.tensor(([0.01 , 0.01, 0.01]),dtype=dtype,device=device)    
    predictionHorizon = 15

    ctrl = mppi.MPPI(dynamics = sys.Dynamcis,
                     running_cost = sys.running_cost,
                     terminal_state_cost = sys.terminal_state_cost,
                     lambda_= 0.001,
                     u_min = umin,
                     u_max = umax,
                     nx    = sys.nx,
                     noise_sigma = noise_sigma,
                     num_samples = 50, 
                     horizon = predictionHorizon,
                     u_scale = 1,
                     device  = device)

    ## Obstacles
    obs1 =  torch.tensor([0.03,0,0.1],device=device)
    obs2 =  torch.tensor([-0.03,0,0.11],device=device)
    sys.updateObsPos(id=1,pos=obs1)
    sys.updateObsPos(id=2,pos=obs2)

    obsPos1 = None
    obsPos2 = None
    
    # neural ode models   
    pr   = prediction("neuralODE/trainedModels/model_NODE.zip")  # 25 points

    q = np.array([0.0, -0.0, -0.0])

    ts = 0.1
    sys.ts = ts
    tf = 50
    gt = 0
    x0 = np.array((0, 0, 0.1))
    endtip = np.array((0, 0, 0.1))
    actions = np.array((0, 0, 0))
    ref = None
    ode = ODE()
    ode.updateAction(q)
    ode.odeStepFull()
    xc = ode.states[:3]

    K = 1*np.diag((2.45, 2.45, 2.45))
    tp = time.time()
    t0 = tp

    timestr = time.strftime("%Y%m%d-%H%M%S")
    logFname = "neuralODE/logs/log_" + timestr+".dat"
    logState = np.array([])

    for i in range(int(tf/ts)):
        t = time.time()
        dt = t - tp
        tp = t

        xd,xd_dot, sys.ref = get_ref(gt,horizon=predictionHorizon,traj='helix')

        if ref is None:
            ref = np.copy(xd)
        else:
            ref = np.vstack((ref, xd))

        if obsPos1 is None:
            obsPos1 = np.copy(sys.obs1.cpu().numpy())
            obsPos2 = np.copy(sys.obs2.cpu().numpy())
        else:
            obsPos1 = np.vstack((obsPos1,sys.obs1.cpu().numpy()))
            obsPos2 = np.vstack((obsPos2,sys.obs2.cpu().numpy()))
           
        with torch.no_grad():
            # Start profiling
            # profiler.enable()

            # jac = pr.Jac(q).T # xdot = J qdot
            jac = pr.MLP_Jac(q).T
            sys.Jac = jac
            # sys.xc = torch.tensor(xc,device=device)
            qdot = ctrl.command(torch.tensor(xc,device = device)).cpu().numpy()
            sys.lastU = qdot

            q += (qdot * ts)
            
            # Stop profiling
            # profiler.disable()
            # Generate the profiling report
            # profiler.print_stats()

            ee = pr.predict(q, True).cpu().numpy()


        if (filteringAct):
            if gt < 0.01:
                qp = np.copy(q)
            q = 0.75*qp + q * 0.3
            qp = np.copy(q)

        ode.updateAction(q)
        ode.odeStepFull()

        if (addNoise):
            mu, sigma = 0, 0.00005
            xc = ode.states[:3]+np.squeeze(np.random.normal(mu, sigma, (1, 3)))
        else:
            xc = ode.states[:3]

        if (filteringObs):
            if gt < 0.01:
                xcp = np.copy(xc)
            xc = 0.8*xcp + xc * 0.2
            xcp = np.copy(xc)

        actions = np.vstack((actions, q))
        endtip = np.vstack((endtip, xc))

        if (saveLog):
            dummyLog = np.concatenate((np.array((gt, dt)), np.squeeze(xc), np.squeeze(xd), np.squeeze(
                xd_dot), np.squeeze(qdot), np.array((q[0], q[1], q[2])), np.squeeze(ee)))
            if logState.shape[0] == 0:
                logState = np.copy(dummyLog)
            else:
                logState = np.vstack((logState, dummyLog))

        gt += ts
        print(f"t:{gt:3.3f}\tdt:{dt:3.3f}")

    if (animEn):
        sfVis = softRobotVisualizer(obsEn=False)
        sfVis.actions = actions
        sfVis.endtips = endtip
        sfVis.speed = 20
        # sfVis.obsPos1 = obsPos1
        # sfVis.obsPos2 = obsPos2
        

        len = int(actions.shape[0]/sfVis.speed)

        sfVis.ax.plot3D(ref[:, 0], ref[:, 1], ref[:, 2],
                        'k--', lw=2, label='Ref')
        sfVis.ax.plot3D(0, 0, 0.1, c='r', lw=2, label='Robot')
        plt.legend()

        ani = animation.FuncAnimation(
            sfVis.fig, sfVis.update_graph, len, interval=100, blit=False)
        plt.show()

        if saveGif:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            gifName = "neuralODE/savedFigs/gif_NodeRedMPPI_" + timestr+".gif"
            print(f"saving gif: {gifName}")

            writergif = animation.PillowWriter(fps=15)
            ani.save(gifName, writer=writergif)
            print(f"gif file has been saved: {gifName}")

    if (saveLog):
        with open(logFname, "w") as txt:  # creates empty file with header
            txt.write("#l,ux,uy,x,y,z\n")

        np.savetxt(logFname, logState, fmt='%.5f')
        print(f"log file has been saved: {logFname}")

        # if (i%10 == 0):
        # plt.plot(t,q[0]+q[1],'r*')
        # plt.plot(t,q[0]-q[1],'g*')
        # plt.plot(t,q[0]+q[2],'b*')
        # plt.plot(t,q[0]-q[2],'k*')
        # plt.pause(0.01)

        # ts = 1
        # plt.plot(i*ts,eep[0],'r*',label = 'dl')
        # plt.plot(i*ts,eep[1],'go',label = 'l1')
        # plt.plot(i*ts,eep[2],'bx',label = 'l2')
        # plt.xlabel ('time (s)')
        # plt.ylabel ('length (m)')

        # visualize(true_y, pred_y, func, ii)

    # plt.show()
