# In[16]:


import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.InteractiveSession()


# In[17]:


#####################  Global variables  ####################
global eps,Ez,num_j,d_time
eps    = 10**-5
Ez     = np.array([0,0,1])
num_j  = 7   # [joints]
d_time = 0.1 # [sec]

### Reward ####
w_a    = 1
w_end  = 5
w_done = 20
k1     = 40
k3     = 20  

#####################  Hyper parameters  ####################
global LR_A, LR_C, GAMMA, TAU, MEMORY_CAPACITY, BATCH_SIZE, std_dev
LR_A  = 0.001    # learning rate for actor
LR_C  = 0.001    # learning rate for critic
GAMMA = 0.9      # reward discount
TAU   = 0.001    # soft replacement
MEMORY_CAPACITY = 80000
BATCH_SIZE = 32
std_dev    = 0.2 # OU_noise


# In[18]:


# A skew-symmetric operator
def tilde(x): 
    w = np.array([
        [    0, -x[2],  x[1]],
        [ x[2],     0, -x[0]],
        [-x[1],  x[0],    0]])
    return w


# In[19]:


# Euler angle rotation 
def Rx(theta):
    return np.matrix([[ 1,    0           ,    0           ],
                       [ 0, math.cos(theta),-math.sin(theta)],
                       [ 0, math.sin(theta), math.cos(theta)]])
  
def Ry(theta):
    return np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                       [    0           , 1,    0           ],
                       [-math.sin(theta), 0, math.cos(theta)]])
  
def Rz(theta):
    return np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                       [ math.sin(theta), math.cos(theta) , 0 ],
                       [    0           ,    0            , 1 ]])


# In[20]:


# Transformation matrix
def TRz(theta):
    return np.matrix([[ math.cos(theta), -math.sin(theta), 0 , 0],
                       [ math.sin(theta), math.cos(theta) , 0 , 0],
                       [    0           ,    0            , 1 , 0],
                       [    0           ,    0            , 0 , 1]])
def Tz(d):
    A        = np.eye(4)
    A[2][3] += d
    return A

def Tx(a):
    A        = np.eye(4)
    A[0][3] += a
    return A

def TRx(theta):
    return np.matrix([[ 1,    0           ,    0            , 0],
                       [ 0, math.cos(theta),-math.sin(theta) , 0],
                       [ 0, math.sin(theta), math.cos(theta) , 0],
                       [ 0,    0           ,    0            , 1]])


# In[21]:


def calc_DH(theta, d, a, alpha):
    return TRz(theta)@Tz(d)@Tx(a)@TRx(alpha)


# In[22]:


def qtn2R(q):
    R = np.eye(3)*q[0]**2
    
    R[0][0] +=  + q[1]**2 - q[2]**2 - q[3]**2 
    
    R[1][1] +=  - q[1]**2 + q[2]**2 - q[3]**2 
    R[2][2] +=  - q[1]**2 - q[2]**2 + q[3]**2 
    
    R[1][0] = 2*(q[1]*q[2] + q[0]*q[3])
    R[2][0] = 2*(q[1]*q[3] - q[0]*q[2])
    R[2][1] = 2*(q[2]*q[3] + q[0]*q[1])
    
    R[0][1] = 2*(q[1]*q[2] - q[0]*q[3])
    R[0][2] = 2*(q[1]*q[3] + q[0]*q[2])
    R[1][2] = 2*(q[2]*q[3] - q[0]*q[1])
    
    return R


# In[23]:


class ArmEnv(object):     
    # Class variables - None     
    
    def __init__(self, MAX_EPISODES=5000, MAX_EP_STEPS=250):    
        # Instance variables
        self.MAX_EPISODES = MAX_EPISODES
        self.MAX_EP_STEPS = MAX_EP_STEPS
        self.Time         = [i*d_time for i in range(self.MAX_EP_STEPS)]   
        self.action_dim   = num_j          # Joints angular vel[rad/s]
        self.done         = False
        self.on_goal      = 0
        self.p_end_print  = np.zeros((MAX_EP_STEPS, 3))
        self.p_tar_print  = np.zeros((MAX_EP_STEPS, 3))
        self.dist_print   = np.zeros(MAX_EP_STEPS)
        self.angl_print   = np.zeros(MAX_EP_STEPS)
        self.angl1_print  = np.zeros(MAX_EP_STEPS)
        self.angl2_print  = np.zeros(MAX_EP_STEPS)
        self.angl3_print  = np.zeros(MAX_EP_STEPS)
        self.angl4_print  = np.zeros(MAX_EP_STEPS)
        self.angl5_print  = np.zeros(MAX_EP_STEPS)
        self.angl6_print  = np.zeros(MAX_EP_STEPS)
        self.angl7_print  = np.zeros(MAX_EP_STEPS)
        self.anglv1_print  = np.zeros(MAX_EP_STEPS)
        self.anglv2_print  = np.zeros(MAX_EP_STEPS)
        self.anglv3_print  = np.zeros(MAX_EP_STEPS)
        self.anglv4_print  = np.zeros(MAX_EP_STEPS)
        self.anglv5_print  = np.zeros(MAX_EP_STEPS)
        self.anglv6_print  = np.zeros(MAX_EP_STEPS)
        self.anglv7_print  = np.zeros(MAX_EP_STEPS)
        self.EEvel_print   = np.zeros(MAX_EP_STEPS)
        Time     = d_time*np.array(range(MAX_EP_STEPS))
        
        self.joint_bound      = np.zeros(num_j+1, dtype = [('L',np.float32),('U',np.float32)])
        self.joint_bound['L'] = [0,-170,-150,-150, 30,-170,-170,-170]
        self.joint_bound['U'] = [0, 170, 150, 150, 150, 170, 170,170]

        self.action_bound   = [ -0.2, 0.2]
        # 0   : Base
        # 1~7 : Joint
        self.body_prop_info = np.zeros(num_j+1, dtype=[('mass', np.float32),('link', np.float32),('Ixx', np.float32),('Iyy',np.float32),('Izz',np.float32)])
        self.body_prop_info['mass']  = [1036.835, 3.691, 60.416, 3.678, 35.653,   3.46, 2.624, 0.113]
        self.body_prop_info['link']  = [       1,  0.13,    1.5,  0.12,   1.24,   0.22,  0.07,  0.01]                                    
        self.body_prop_info['Ixx']   = [   291.1, 0.013, 13.556, 0.012,  1.235, 0.0619, 0.006, 2.55*eps]
        self.body_prop_info['Iyy']   = [   536.3, 0.005,  2.274,  0.01,  7.309,  0.028, 0.004, 2.55*eps]
        self.body_prop_info['Izz']   = [   669.6, 0.015, 15.756, 0.005,  6.104,  0.035, 0.004, 4.92*eps]

        self.Mass   = sum(self.body_prop_info['mass']) # Arm + Base mass
        X           = self.body_prop_info['link'] 
        self.d_list = [X[1], 0, X[2]+X[3], 0, X[4]+X[5], 0, X[6]+X[7]] # d_bs, d_se, d_ew, d_wt

        # 0 ~ 7 : Joint -> p, z
        # 0     : Base  -> r0
        # 1 ~ 7 : Link  -> r, v, w
        self.Arm_kin_info = np.zeros(num_j+1, dtype=[('theta', np.float32)]) 
        self.State_info   = np.zeros((num_j+1,3),dtype=[('p', np.float32),('r', np.float32),('z', np.float32),('v', np.float32),('w', np.float32)])
        self.Vector_info  = np.zeros(3,dtype=[('rg', np.float32)])

        # 0 ~ 7 : Joints (0 ~ 7)
        # -1    : Base
        self.AA = np.zeros((num_j+2,3,3), dtype=[('A',np.float32)])     # Rotation matrices
        self.AA['A'][0]      = np.eye(3)
        self.AA['A'][-1]     = np.eye(3)
        self.Transf = np.zeros((num_j+2,4,4), dtype=[('T',np.float32)]) # Transformation matrices
        self.Transf['T'][0]  = np.eye(4)
        self.Transf['T'][-1] = np.eye(4)

        # 0 ~ 6 : Joints (1 ~ 7)
        self.J_info      = np.zeros((num_j,3,num_j),dtype=[('Jp', np.float32),('JH',np.float32)])
        self.Jp_info     = np.zeros((3,num_j),dtype=[('JpPhi', np.float32),('Pphi', np.float32)])

        # Target information ( Starlink )
        self.Target_info = np.zeros(3, dtype=[('r', np.float32),('h_w_radius', np.float32)]) # Workspace needs to be specified
        self.Target_info['r']          = [ 0,    1,   1]
        self.Target_info['h_w_radius'] = [ 1, 0.05, 0.5]

        self.target = {'l': 0.1 } # Goal area 
        self.beta   = math.atan2(self.Target_info['h_w_radius'][2],self.Target_info['h_w_radius'][0])

        # Reward variables
        self.ini_dist  = 1

        # DDPG variables
        self.state_dim = 43
        
        # Visualization variables
        self.count_step = 0    
        
    def step(self, action):
        self.count_step  += 1
        self.done         = False
        action            = np.clip(action, *self.action_bound)  
        # Transformation
        # DH-parameter       
        T    = np.zeros((num_j+1,4,4))
        T[0] = np.eye(4)
        T[0,:3,:3] = self.AA['A'][-1]
        T[0,:3, 3] = self.AA['A'][-1]@np.array([ 0, 0, 1]) + self.State_info['r'][0]
        
        for i in range(1,num_j+1):
            alpha = (-1)**i*math.pi/2 if i < num_j else 0
            T[i]  = calc_DH( self.Arm_kin_info['theta'][i], self.d_list[i-1], 0, alpha)
                    
        self.Transf['T'][0] = T[0]           
        for i in range(1,num_j+1):
            self.Transf['T'][i] = self.Transf['T'][i-1]@T[i]
                            
        self.State_info['p'] = self.Transf['T'][:8,:3, 3]
        self.State_info['z'] = self.Transf['T'][:8,:3, 2]
        self.AA['A'][1:]     = self.Transf['T'][1:,:3,:3]
        
        X = self.State_info['p']
        for j in range(1,num_j):
            self.State_info['r'][j] = (X[j+1] + X[j])/2
       
        # GJM
        # 1. Calculate the inertia matrix  
        self.Vector_info['rg'] = self.body_prop_info['mass']@self.State_info['r']/self.Mass
        for i in range(1,num_j+1):
            X = np.zeros((num_j,3))
            for j in range(1,i+1):
                X[j-1][:] = np.array([np.cross(self.State_info['z'][i], self.State_info['r'][i] - self.State_info['p'][j])])
            self.J_info['Jp'][i-1] = X.T
            
        self.Jp_info['JpPhi'] = np.zeros((3,num_j))
        for i in range(1,num_j+1):
            self.Jp_info['JpPhi'] += self.body_prop_info['mass'][i]*np.sum(self.J_info['Jp'][:i], axis=0)
                
        X = np.zeros((num_j,3))
        for i in range(num_j):
            X[i][:] = self.State_info['z'][i+1]
            self.J_info['JH'][i] = X.T
            
        self.Jp_info['Pphi'] = np.zeros((3,num_j))
        for i in range(1,num_j+1):
            I_matrix = np.zeros((3,3))
            I_matrix[0][0] = self.body_prop_info['Ixx'][i]
            I_matrix[1][1] = self.body_prop_info['Iyy'][i]
            I_matrix[2][2] = self.body_prop_info['Izz'][i]
            self.Jp_info['Pphi'] += I_matrix.dot(self.J_info['JH'][i-1]) + self.body_prop_info['mass'][i]*tilde(self.State_info['r'][i]).dot(self.J_info['Jp'][i-1])
            
        Pw = np.zeros((3,3))
        Pw[0][0] = sum(self.body_prop_info['Ixx'])
        Pw[1][1] = sum(self.body_prop_info['Iyy'])
        Pw[2][2] = sum(self.body_prop_info['Izz'])

        for i in range(1,num_j+1):
            Pw -= self.body_prop_info['mass'][i]*tilde(self.State_info['r'][i]).dot(tilde(self.State_info['r'][i] - self.State_info['r'][0]))    
        
        X    = self.Mass*np.concatenate((np.eye(3), tilde(self.Vector_info['rg'])),axis = 0)     # 세로 합치기
        Y    = np.concatenate((-self.Mass*tilde(self.Vector_info['rg'] - self.State_info['r'][0]),Pw),axis = 0)  # 세로 합치기
        Hs   = np.concatenate((X,Y),axis = 1)                                              # 가로 합치기 
        Hm   = np.concatenate((self.Jp_info['JpPhi'],self.Jp_info['Pphi']),axis = 0)       # 세로 합치기
        E12  = -LA.inv(Hs).dot(Hm)
        
        p0ex = tilde(self.State_info['p'][-1] - self.State_info['r'][0])
        Jm   = np.concatenate((self.J_info['Jp'][-1],self.J_info['JH'][-1]),axis = 0)   
        X    = np.concatenate((np.eye(3),np.zeros((3,3))), axis = 0)  # 세로 합치기
        Y    = np.concatenate((-p0ex,np.eye(3)),axis = 0)             # 세로 합치기
        J0   = np.concatenate((X,Y),axis = 1)                         # 가로 합치기
        
        GJ   = Jm + J0.dot(E12)
        
        v_w_0 = E12@action
        v_w_e =  GJ@action       
        
        ## State update 
        # Flight dynamics
        r_temp = self.State_info['r']
        v_temp = self.State_info['v']
        w_temp = self.State_info['w']
        w0     = w_temp[0]
        
        self.State_info['r'][0] += v_temp[0]*d_time
        for i in range(1,num_j):
            self.State_info['v'][i] = v_temp[i] + np.cross( w_temp[0], r_temp[i] - r_temp[0] ) + np.sum(np.cross(self.State_info['z'][:i],r_temp[i] - self.State_info['p'][:i]),axis=0)  
        
        self.State_info['v'][0]  = v_w_0[:3]
        self.State_info['w'][0]  = v_w_0[3:6]
        self.State_info['v'][-1] = v_w_e[:3]
        self.State_info['w'][-1] = v_w_e[3:6]
        
        # Joint 1 ~ 6 angular vel 구하기
        Norm_w = np.zeros(num_j)
        for i in range(1,num_j):
            self.State_info['w'][i] = self.State_info['z'][i]*action[i-1]
            Norm_w[i-1] = LA.norm(self.State_info['w'][i])
                    
        self.Arm_kin_info['theta'][1:] += action*d_time
        self.Arm_kin_info['theta'][0]   = LA.norm(w_temp[0])*d_time       
                        
        if LA.norm(w0) == 0:
            self.AA['A'][-1] = np.eye(3)
        else:
            th = LA.norm(w0)*d_time
            w  = w0 / LA.norm(w0) 
            c  = math.cos(th)
            s  = math.sin(th)
            self.AA['A'][-1] = np.array([[c + w[0]**2*(1-c),     w[0]*w[1]*(1-c)-w[2]*s, w[2]*w[0]*(1-c)+w[1]*s],
                                           [w[0]*w[1]*(1-c)+w[2]*s,        c+w[1]**2*(1-c), w[2]*w[1]*(1-c)-w[0]*s],
                                           [w[2]*w[0]*(1-c)-w[1]*s, w[2]*w[1]*(1-c)+w[0]*s,       c+w[2]**2*(1-c)]])    
        # Update target rotation
        self.Target_info['r'] = np.array([ 0, 1, 2])
        X     = self.Target_info['h_w_radius']
        alpha = (X[1]*d_time*self.count_step)%(2*math.pi)
        self.Target_info['r'][0] += X[-1]*math.sin(alpha)
        self.Target_info['r'][1] -= X[-1]*math.cos(alpha)
         
        X = np.array([math.cos(alpha/2)])
        Y = Ez*math.sin(alpha/2)
        q = np.concatenate((X,Y), axis = 0)
        
        target_A = qtn2R(q)@Rx(self.beta)
                
        # Capturing Distance
        dist = LA.norm(self.Target_info['r'] - self.State_info['p'][-1])
        
        # Capturing Angle
        # Target y-axis 
        # End-effector y-axis 
        target_y = np.zeros(3)
        for i in range(3):
            target_y[i] = target_A[i,1]
        endeff_y = self.AA['A'][7,:,1]
        
        innerAB  = target_y@endeff_y
        AB       = LA.norm(target_y)*LA.norm(endeff_y)
        angle_e  = np.arccos(innerAB/AB) # [rad]
        
        r = - math.log10(dist + eps)
        if dist < 0.4:
            r += w_done
        if angle_e < 0.1:
            r += w_done
        if dist < 0.4 and angle_e < 0.1:    
            self.done = True
            
        # State
        s = np.append(self.State_info['r'][0], self.AA['A'][-1].reshape(1,-1))
        s = np.append(s,self.State_info['v'][0])
        s = np.append(s,self.Arm_kin_info['theta'])
        s = np.append(s,LA.norm(self.State_info['w'][0]))
        s = np.append(s,Norm_w)
        s = np.append(s,self.State_info['p'][-1])
        s = np.append(s,self.State_info['v'][-1])
        s = np.append(s,self.Target_info['r'])
        s = np.append(s,self.Target_info['h_w_radius'][1]*self.Target_info['h_w_radius'][2])
        s = np.append(s,dist)
        s = np.append(s,angle_e)
        
        # Visualization
        self.p_end_print[self.count_step-1] = self.State_info['p'][-1]
        self.p_tar_print[self.count_step-1] = self.Target_info['r']
        self.dist_print[self.count_step-1]  = dist
        self.angl_print[self.count_step-1]  = angle_e          # [rad]
        self.angl1_print[self.count_step-1]  = self.Arm_kin_info['theta'][1]
        self.angl2_print[self.count_step-1]  = self.Arm_kin_info['theta'][2]
        self.angl3_print[self.count_step-1]  = self.Arm_kin_info['theta'][3]
        self.angl4_print[self.count_step-1]  = self.Arm_kin_info['theta'][4]
        self.angl5_print[self.count_step-1]  = self.Arm_kin_info['theta'][5]
        self.angl6_print[self.count_step-1]  = self.Arm_kin_info['theta'][6]
        self.angl7_print[self.count_step-1]  = self.Arm_kin_info['theta'][7]
        
        self.anglv1_print[self.count_step-1]  = LA.norm(Norm_w[0])
        self.anglv2_print[self.count_step-1]  = LA.norm(Norm_w[1])
        self.anglv3_print[self.count_step-1]  = LA.norm(Norm_w[2])
        self.anglv4_print[self.count_step-1]  = LA.norm(Norm_w[3])
        self.anglv5_print[self.count_step-1]  = LA.norm(Norm_w[4])
        self.anglv6_print[self.count_step-1]  = LA.norm(Norm_w[5])
        self.anglv7_print[self.count_step-1]  = LA.norm(Norm_w[6])
        self.reward_print[self.count_step-1]  = r
        self.EEvel_print[self.count_step-1]   = LA.norm(self.State_info['v'][-1])
        return s, r, self.done
    
    def reset(self):
        #self.Arm_kin_info['theta'] = [ 0, 0, 30*math.pi/180, 0, 30*math.pi/180, 0, 60*math.pi/180, 0]
        self.Arm_kin_info['theta'] = [ 0, 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random()]
        
        self.State_info['r'][0] = 0
        self.State_info['v']    = np.zeros((num_j+1,3))
        self.State_info['w']    = np.zeros((num_j+1,3))
        self.AA['A'][-1]        = np.eye(3)
        self.Transf['T'][-1]    = np.eye(4)
        
        # 0 ~ 6 : Joints(1 ~ 7)
        self.J_info      = np.zeros((num_j,3,num_j),dtype=[('Jp', np.float32),('JH',np.float32)])
        self.Jp_info     = np.zeros((3,num_j),dtype=[('JpPhi', np.float32),('Pphi', np.float32)])
                
        # Transformation
        # DH - parameter       
        T    = np.ones((num_j+1, 4, 4))
        T[0] = np.eye(4)
        T[0,:3,:3] = self.AA['A'][-1]
        T[0,:3, 3] = self.AA['A'][-1]@np.array([ 0, 0, 1]) + self.State_info['r'][0]
        
        for i in range(1,num_j+1):
            alpha = (-1)**i*math.pi/2 if i < num_j else 0    
            T[i]  = calc_DH(self.Arm_kin_info['theta'][i], self.d_list[i-1], 0, alpha)
                        
        self.Transf['T'][0] = T[0]           
        for i in range(1,num_j+1):
            self.Transf['T'][i] = self.Transf['T'][i-1]@T[i]
                
        self.State_info['p'] = self.Transf['T'][:8,:3,3]
        self.State_info['z'] = self.Transf['T'][:8,:3,2]
        self.AA['A'][1:]     = self.Transf['T'][1:,:3,:3]
            
        X = self.State_info['p']
        for j in range(1,num_j):
            self.State_info['r'][j] = (X[j+1] + X[j])/2
       
        # GJM
        # 1. Calculate the inertia matrix  
        self.Vector_info['rg'] = self.body_prop_info['mass']@self.State_info['r']/self.Mass    
        for i in range(1,num_j+1):
            X = np.zeros((num_j,3))
            for j in range(1,i+1):
                X[j-1][:] = np.array([np.cross(self.State_info['z'][i], self.State_info['r'][i] - self.State_info['p'][j])])
            self.J_info['Jp'][j-1] = X.T
        for i in range(1,num_j+1):
            a = self.body_prop_info['mass'][i]*np.sum(self.J_info['Jp'][:i], axis = 0)
            self.Jp_info['JpPhi'] += self.body_prop_info['mass'][i]*np.sum(self.J_info['Jp'][:i], axis = 0)
        
        X = np.zeros((num_j,3))
        for i in range(num_j):
            X[i][:] = self.State_info['z'][i+1]
            self.J_info['JH'][i] = X.T
            
        for i in range(1,num_j+1):
            I_matrix       = np.zeros((3,3))
            I_matrix[0][0] = self.body_prop_info['Ixx'][i]
            I_matrix[1][1] = self.body_prop_info['Iyy'][i]
            I_matrix[2][2] = self.body_prop_info['Izz'][i]
            self.Jp_info['Pphi'] += I_matrix.dot(self.J_info['JH'][i-1]) + self.body_prop_info['mass'][i]*tilde(self.State_info['r'][i]).dot(self.J_info['Jp'][i-1])
                        
        Pw       = np.zeros((3,3))
        Pw[0][0] = sum(self.body_prop_info['Ixx'])
        Pw[1][1] = sum(self.body_prop_info['Iyy'])
        Pw[2][2] = sum(self.body_prop_info['Izz'])
                
        for i in range(1,num_j+1):
            Pw -= self.body_prop_info['mass'][i]*tilde(self.State_info['r'][i]).dot(tilde(self.State_info['r'][i] - self.State_info['r'][0]))    
        
        X    = self.Mass*np.concatenate(( np.eye(3), tilde(self.Vector_info['rg'])), axis = 0)                    # 세로 합치기
        Y    = np.concatenate((-self.Mass*tilde(self.Vector_info['rg'] - self.State_info['r'][0]),Pw), axis = 0)  # 세로 합치기
        Hs   = np.concatenate((X,Y), axis = 1)                                        # 가로 합치기 
        Hm   = np.concatenate((self.Jp_info['JpPhi'],self.Jp_info['Pphi']), axis = 0) # 세로 합치기
        E12  = -LA.inv(Hs).dot(Hm)
        
        p0ex = tilde(self.State_info['p'][-1] - self.State_info['r'][0])
        Jm   = np.concatenate((self.J_info['Jp'][-1],self.J_info['JH'][-1]),axis = 0)   
        X    = np.concatenate((np.eye(3),np.zeros((3,3))), axis = 0)# 세로 합치기
        Y    = np.concatenate((-p0ex,np.eye(3)),axis = 0)           # 세로 합치기
        J0   = np.concatenate((X,Y),axis = 1)                       # 가로 합치기
        
        GJ   = Jm + J0.dot(E12)
        
        v_w_0 = np.zeros(6)
        v_w_e = np.zeros(6)      
        
        ## State update ##                
        # Update target rotation
        self.Target_info['r'] = np.array([ 0, 1, 2])
        X     = self.Target_info['h_w_radius']
        alpha = 0
    
        self.Target_info['r'][0] += X[-1]*math.sin(alpha)
        self.Target_info['r'][1] -= X[-1]*math.cos(alpha)
         
        X = np.array([math.cos(alpha/2)])
        Y = Ez*math.sin(alpha/2)
        q = np.concatenate((X,Y), axis=0)
                
        target_A = qtn2R(q)@Rx(self.beta)
        # Capturing distance
        self.ini_dist = LA.norm(self.Target_info['r'] - self.State_info['p'][-1])
       
        # Capturing angle
        # Target y-axis 
        # End-effector y-axis
        target_y = np.zeros(3)
        for i in range(3):
            target_y[i] = target_A[i,1]
        endeff_y = self.AA['A'][7,:,1]
        
        innerAB  = target_y@endeff_y
        AB       = LA.norm(target_y)*LA.norm(endeff_y)
        angle_e  = np.arccos(innerAB/AB) # [rad]
        
        # State
        s = np.append(self.State_info['r'][0], self.AA['A'][-1].reshape(1,-1))
        s = np.append(s,self.State_info['v'][0])
        s = np.append(s,self.Arm_kin_info['theta'])
        s = np.append(s,np.zeros(num_j+1))
        s = np.append(s,self.State_info['p'][-1])
        s = np.append(s,self.State_info['v'][-1])
        s = np.append(s,self.Target_info['r'])
        s = np.append(s,self.Target_info['h_w_radius'][1]*self.Target_info['h_w_radius'][2])
        s = np.append(s,self.ini_dist)
        s = np.append(s,angle_e)
        
        # Visualization
        self.count_step  = 0
        self.on_goal     = 0
        self.p_end_print = np.zeros((MAX_EP_STEPS, 3))
        self.p_tar_print = np.zeros((MAX_EP_STEPS, 3))
        self.dist_print  = np.zeros(MAX_EP_STEPS)
        self.angl_print  = np.zeros(MAX_EP_STEPS)
        self.reward_print = np.zeros(MAX_EP_STEPS)        
        self.angl1_print  = np.zeros(MAX_EP_STEPS)
        self.angl2_print  = np.zeros(MAX_EP_STEPS)
        self.angl3_print  = np.zeros(MAX_EP_STEPS)
        self.angl4_print  = np.zeros(MAX_EP_STEPS)
        self.angl5_print  = np.zeros(MAX_EP_STEPS)
        self.angl6_print  = np.zeros(MAX_EP_STEPS)
        self.angl7_print  = np.zeros(MAX_EP_STEPS)
        self.anglv1_print  = np.zeros(MAX_EP_STEPS)
        self.anglv2_print  = np.zeros(MAX_EP_STEPS)
        self.anglv3_print  = np.zeros(MAX_EP_STEPS)
        self.anglv4_print  = np.zeros(MAX_EP_STEPS)
        self.anglv5_print  = np.zeros(MAX_EP_STEPS)
        self.anglv6_print  = np.zeros(MAX_EP_STEPS)
        self.anglv7_print  = np.zeros(MAX_EP_STEPS)
        self.EEvel_print   = np.zeros(MAX_EP_STEPS)
        return s
    
    def print_figures(self,MAX_EP_STEPS):
        fig = plt.figure(figsize=(10,10))    
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.p_end_print[:,0], self.p_end_print[:,1], self.p_end_print[:,2], c= "red")
        ax.scatter(self.p_tar_print[:,0], self.p_tar_print[:,1], self.p_tar_print[:,2], c="blue")
        
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(["Positions of End Effector(red) & Target(blue)"])         
        
        plt.figure(figsize=(6,6))
        plt.plot(self.Time, self.dist_print)
        plt.title('Distance[m] VS Time[sec]')
        
        plt.figure(figsize=(6,6))
        plt.plot(self.Time, self.angl_print)
        plt.title('Angle error[rad] VS Time[sec]')
        
        # Joint angle
        plt.figure(figsize=(6,6))
        plt.plot(self.Time, self.angl1_print,'r')
        plt.plot(self.Time, self.angl2_print,'g')
        plt.plot(self.Time, self.angl3_print,'b')
        plt.plot(self.Time, self.angl4_print,'c')
        plt.plot(self.Time, self.angl5_print,'m')
        plt.plot(self.Time, self.angl6_print,'y')
        plt.plot(self.Time, self.angl7_print,'k')
        plt.title('Angle[rad] VS Time[sec]')
        
        # Angular velocity
        plt.figure(figsize=(6,6))
        plt.plot(self.Time, self.anglv1_print,'r')
        plt.plot(self.Time, self.anglv2_print,'g')
        plt.plot(self.Time, self.anglv3_print,'b')
        plt.plot(self.Time, self.anglv4_print,'c')
        plt.plot(self.Time, self.anglv5_print,'m')
        plt.plot(self.Time, self.anglv6_print,'y')
        plt.plot(self.Time, self.anglv7_print,'k')
        plt.title('Angular Velocity[rad/s] VS Time[sec]')
        
        # End-effector velocity
        plt.figure(figsize=(6,6))
        plt.plot(self.Time, self.EEvel_print,'g')
        plt.title('End-effector Velocity[m/s] VS Time[sec]')
        
        plt.figure(figsize=(6,6))
        plt.plot(self.Time, self.reward_print)
        plt.title('Reward VS Time[sec]')

        plt.show()
                
    def render(self):
        pass


# In[24]:


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory  = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]
        self.S  = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R  = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor', reuse=tf.AUTO_REUSE):
            self.a = self._build_a(self.S, scope='eval',     trainable=True)  
            a_     = self._build_a(self.S_, scope='target' , trainable=False)
            
        with tf.variable_scope('Critic', reuse=tf.AUTO_REUSE):
            # Assign self.a = a in memory when calculating q for td_error,
            # Otherwise the self.a is from Actor when updating Actor            
            q  = self._build_c(self.S, self.a, scope='eval', trainable= True)
            q_ = self._build_c(self.S_, a_, scope='target' , trainable=False)

        # Networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # Target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]
        q_target    = self.R + GAMMA * q_
        
        # In the feed_dic for the td_error, the self.a should change to actions in memory
        td_error    = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss      = - tf.reduce_mean(q)    # Maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[None, :]})[0]

    def learn(self):
        # Soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:,:self.s_dim]
        ba = bt[:,self.s_dim: self.s_dim + self.a_dim]
        br = bt[:,-self.s_dim - 1: -self.s_dim]
        bs_= bt[:,-self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition    = np.hstack((s, a, [r], s_))
        index         = self.pointer % MEMORY_CAPACITY  # Replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:              # Indicator for learning
            self.memory_full = True

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(s, 200, activation=tf.nn.relu, name='l1', trainable=trainable)
            a   = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            n_l1 = 200
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1   = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net  = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
    
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params')


# In[25]:


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.randint(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return

        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.observations0)


# In[26]:


class ActionNoise(object):
    def reset(self):
        pass

#Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# In[27]:


# set env
tf.reset_default_graph()
env      = ArmEnv()
ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=float(std_dev) * np.ones(1))
s_dim    = env.state_dim
a_dim    = env.action_dim
a_bound  = [-1, 1]

# set RL method (continuous)
rl         = DDPG(a_dim, s_dim, a_bound)
steps      = []
reward_plt = []
success    = []
distance   = []

MAX_EPISODES = 5000
MAX_EP_STEPS = 250

def train():
    # Start training
    for i in range(MAX_EPISODES):       
        s        = env.reset()
        ep_r     = 0
        ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=float(std_dev) * np.ones(1)) # decreasing noise
        ep_step  = 0
        for j in range(MAX_EP_STEPS):
            a = rl.choose_action(s) + ou_noise() 
            s_, r, done = env.step(a)
            rl.store_transition(s, a, r, s_)
            ep_r += r
                
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j+1))
                reward_plt.append(ep_r)
                ep_step += 1
                if done:
                    success.append(1)
                else:
                    success.append(0)
                    
                print('Error angle: %.1f' %(s[-1]))
                distance.append(s[-2])   
                break
            ep_step += 1
        #if done :
        env.print_figures(ep_step)
    rl.save()

def eval():
    rl.restore()
    s = env.reset()
    while True:
        a = rl.choose_action(s)
        s, r, done = env.step(a)
    
train()
success_rate = []
for i in range(int(len(success)/100)):
    summ = 0
    for j in range(100):
        summ += success[j + i*100]
    success_rate.append(summ/100)

xaxis = np.arange(len(reward_plt))

plt.figure()
plt.subplot(211)
plt.plot(xaxis,np.array(reward_plt))
plt.title("Rewards per Episode")

plt.subplot(212)
plt.plot(np.arange(len(success_rate)),np.array(success_rate))
plt.title("success rate")

plt.show()
