import pybullet as p
import pybullet_data
import numpy as np
import math

from visualizer.visualizer import ODE
from scipy.spatial.transform import Rotation as Rot


class SoftRobotBasicEnvironment():
    def __init__(self) -> None:
        self._simulationStepTime = 0.005
        self.vis = True

        p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self._simulationStepTime)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(cameraDistance=0.3, cameraYaw=180, cameraPitch=-35, cameraTargetPosition=[0.,0,0.1])
        self._pybullet = p
        self.plane_id = p.loadURDF('plane.urdf')
        self._ode = ODE()
        self.create_robot()


    def _dummy_sim_step(self,n):
        for _ in range(n):
            p.stepSimulation()


  
    def add_a_cube_without_collision(self,pos,size=[0.1,0.1,0.1], color = [0.1,0.1,0.1,1],textureUniqueId = None):

        # cubesID = []
        box     = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        vis     = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=color)
        obj_id  = p.createMultiBody(0, box, vis, pos, [0,0,0,1])
        p.stepSimulation()
        if textureUniqueId is not None:
            p.changeVisualShape(obj_id, -1, textureUniqueId=textureUniqueId)
        return obj_id 
    
    def add_a_cube(self,pos,size=[0.1,0.1,0.1],mass = 0.1, color = [1,1,0,1], textureUniqueId = None):

        # cubesID = []
        box     = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        vis     = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=color)
        obj_id  = p.createMultiBody(mass, box, vis, pos, [0,0,0,1])
        p.changeDynamics(obj_id, 
                        -1,
                        spinningFriction=800,
                        rollingFriction=0.0,
                        linearDamping=50.0)
        
        if textureUniqueId is not None:
            p.changeVisualShape(obj_id, -1, textureUniqueId=textureUniqueId)

        # cubesID.append(obj_id)
        
        p.stepSimulation()
        return obj_id 
    
    def calculate_orientation(self, point1, point2):
        # Calculate the difference vector
        diff = np.array(point2) - np.array(point1)
        
        # Calculate yaw (around z-axis)
        yaw = math.atan2(diff[1], diff[0])
        
        # Calculate pitch (around y-axis)
        pitch = math.atan2(-diff[2], math.sqrt(diff[0]**2 + diff[1]**2))

        # Roll is arbitrary in this context, setting it to zero
        roll = 0

        return p.getQuaternionFromEuler([roll, pitch, yaw])


    def create_robot(self,number_of_sphere = 20,color = [0.6, .6, 0.6, 1],body_base_color = [0.3,0.3,0.3,1], body_base_leg_color = [0.8,0.8,0.8,1]):        
        
        act = np.array([0,0,0.0])
        self._ode.updateAction(act)
        sol = self._ode.odeStepFull()
        
        self._base_pos = np.array([0,0,0.1])
        texUid = p.loadTexture("pybullet_env/textures/table_tecture.png")
        self.add_a_cube_without_collision(pos = [-0.,0.,0],size=[0.5,0.5,0.01],color=[0.7,0.7,0.7,1],textureUniqueId=texUid) # table
        
        self.add_a_cube_without_collision(pos = [ 0.,-0.1,0.1],size=[0.1,0.2,0.1],color=body_base_color) # body
        self.add_a_cube_without_collision(pos = [ 0.041,-0.009,0.05],size=[0.02,0.02,0.1],color=body_base_leg_color) #legs
        self.add_a_cube_without_collision(pos = [-0.041,-0.009,0.05],size=[0.02,0.02,0.1],color=body_base_leg_color)

        self.add_a_cube_without_collision(pos = [ 0.041,-0.189,0.05],size=[0.02,0.02,0.1],color=body_base_leg_color)
        self.add_a_cube_without_collision(pos = [-0.041,-0.189,0.05],size=[0.02,0.02,0.1],color=body_base_leg_color)
        
        # Define the shape and color parameters (change these as needed)
        radius = 0.01
        self._number_of_sphere = number_of_sphere
        
        shape = p.createCollisionShape(p.GEOM_SPHERE,  radius=radius)
        visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        
        visualShapeId_tip = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02,0.002,0.001], rgbaColor=[1,0,0,1])
        visualShapeId_tip_ = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.005,  rgbaColor=[0.,0,0.75,1])
        
        # Load the positions
        idx = np.linspace(0,sol.shape[1]-1,self._number_of_sphere,dtype=int)
        positions = [(sol[0,i], sol[2,i], sol[1,i]) for i in idx]
       
        # Create a body at each position
        self._robot_bodies = [p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                            baseVisualShapeIndex=visualShapeId,
                            basePosition=pos+self._base_pos) for pos in positions]
        
        ori = self.calculate_orientation (positions[-2],positions[-1])
        self._robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                            baseVisualShapeIndex=visualShapeId_tip_,
                            basePosition=positions[-1]+self._base_pos,
                            baseOrientation= ori))
                                
        self._robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                            baseVisualShapeIndex=visualShapeId_tip,
                            basePosition=positions[-1]+self._base_pos,baseOrientation= ori))
        
        
        
        self._robot_line_ids = []
        # for i, pos in enumerate(positions):
        #     p.resetBasePositionAndOrientation(self._robot_bodies[i], pos, (0, 0, 0, 1))
        #     # Draw a line to the next body
        #     if i < len(positions) - 1:
        #         line_id = p.addUserDebugLine(pos, positions[i + 1], [1, 0, 0])
        #         self._robot_line_ids.append(line_id)

        # if i < len(positions) - 1:
        #     line_id = p.addUserDebugLine(pos, positions[i + 1], [1, 0, 0],lineWidth=2,lifeTime = 0.1)
        #     line_ids.append(line_id)

        self._dummy_sim_step(1)

    
    def move_robot(self,action = np.array([0,0,0]),vis = True):
        self._ode.updateAction(action)
        sol = self._ode.odeStepFull()
        if vis:
            self.visulize(sol)
        

    def visulize (self,sol):
        idx = np.linspace(0,sol.shape[1]-1,self._number_of_sphere,dtype=int)
        positions = [(sol[0,i], sol[2,i], sol[1,i]) for i in idx]
        # for line_id in self._robot_line_ids:
        #     p.removeUserDebugItem(line_id)
        self._robot_line_ids = []
        for i, pos in enumerate(positions):
            p.resetBasePositionAndOrientation(self._robot_bodies[i], pos+self._base_pos, (0, 0, 0, 1))

            # Draw a line to the next body
            # if i < len(positions) - 1:
            #     line_id = p.addUserDebugLine(pos, positions[i + 1], [1, 0, 0],lineWidth = 5,lifeTime=0.4)
            #     self._robot_line_ids.append(line_id)

            # self._dummy_sim_step(1)
            
        
        
        ori = self.calculate_orientation (positions[-2],positions[-1])
        p.resetBasePositionAndOrientation(self._robot_bodies[-2], positions[-1]+self._base_pos, ori)
        p.resetBasePositionAndOrientation(self._robot_bodies[-1], positions[-1]+self._base_pos, ori)
        
        self._dummy_sim_step(100)
        

    def wait(self,sec):
        for _ in range(1+int(sec/self._simulationStepTime)):
            p.stepSimulation()
