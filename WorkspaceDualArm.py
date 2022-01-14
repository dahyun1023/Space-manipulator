import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA


# In[9]:


#####################  Global variables  ####################
global eps,Ez,num_j,d_time
eps      = 10**-6
Ez       = np.array([0,0,1])
num_j    = 6   # [joints]


# In[10]:


# A skew-symmetric operator
def tilde(x): 
    w = np.array([
        [    0, -x[2],  x[1]],
        [ x[2],     0, -x[0]],
        [-x[1],  x[0],    0]])
    return w


# In[11]:


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


# In[12]:


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


# In[13]:


def calc_DH(theta, d, a, alpha):
    return TRz(theta)@Tz(d)@Tx(a)@TRx(alpha)


# In[14]:


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

# In[19]:


j1   = np.linspace( -20, 20,10)*math.pi/180   # theta[deg]
j2   = np.linspace(  90,180,15)*math.pi/180   # alpha[deg]
j3   = np.linspace(-100,100,21)*math.pi/180   # theta[deg]
j4   = np.linspace(  90,180,10)*math.pi/180   # alpha[deg]
j5   = np.linspace( -80, 80,9)*math.pi/180   # theta[deg]
j6   = np.linspace( -80, 80,9)*math.pi/180   # theta[deg]
link = [ 0.16, 0.06,-0.13,-0.06, 0.13, 0.04 ] #  length[m]


# In[20]:


Transf     = np.zeros((num_j+2, 4, 4))
fig        = plt.figure(figsize = (6, 6))
count_step = 0
end_posx   = []
end_posy   = []
end_posz   = []

for a in j1:
    for b in j2:
        for c in j3:
            for d in j4:
                for e in j5:
                    for f in j6:
                        theta = [a, b, c, d, e, f]
                        T     = np.zeros((num_j+1,4,4))                       
                        T[0]  = np.eye(4)
                        T[1]  = calc_DH( theta[0], link[0], 0, 0)
                        T[2]  = calc_DH( 0, link[1], 0, theta[1])
                        T[3]  = calc_DH( theta[2], link[2], 0, 0)
                        T[4]  = calc_DH( 0, link[3], 0, theta[3])
                        T[5]  = calc_DH( theta[4], link[4], 0, 0)
                        T[6]  = calc_DH( theta[5], link[5], 0, 0)
    
                        Transf[0] = T[0]          
                        for i in range(1,num_j+1):
                            Transf[i] = Transf[i-1]@T[i]
            
                        end_posx = np.append(end_posx,Transf[6,0,3])
                        end_posy = np.append(end_posy,Transf[6,1,3])
                        end_posz = np.append(end_posz,Transf[6,2,3])
                        count_step += 1
print("Steps",count_step)


# In[29]:


import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
from numpy.random import randn
from scipy import array, newaxis
get_ipython().run_line_magic('matplotlib', 'notebook')

# ======

Xs = end_posx
Ys = end_posy
Zs = end_posz

# ======
## plot:

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title(["Positions of End Effector"])   

fig.tight_layout()

plt.show()
