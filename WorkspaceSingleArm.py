import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA

# In[5]:

#####################  Global variables  ####################
global eps,Ez,num_j,d_time
eps      = 10**-5
Ez       = np.array([0,0,1])
num_j    = 7   # [joints]

# In[6]:

# A skew-symmetric operator
def tilde(x): 
    w = np.array([
        [    0, -x[2],  x[1]],
        [ x[2],     0, -x[0]],
        [-x[1],  x[0],    0]])
    return w

# In[7]:

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


# In[8]:

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

# In[9]:

def calc_DH(theta, d, a, alpha):
    return TRz(theta)@Tz(d)@Tx(a)@TRx(alpha)

# In[10]:

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

# In[20]:

j1 = np.linspace(0,30,10)
j2 = np.linspace(-150,150,18)
j3 = np.linspace(-150,150,18)
j4 = np.linspace(30,150,20)
j5 = np.linspace(-170,170,20)
j6 = np.linspace(-170,170,20)
j7 = np.linspace(-170,170,20)
link = [  1,  0.13, 1.5,  0.12,   1.24,   0.22,  0.07,  0.01] 


# In[21]:


Transf  = np.zeros((num_j+2, 4, 4))
fig     = plt.figure(figsize = (6, 6))
count_step = 0
end_posx = []
end_posy = []
end_posz = []

for a in j1:
    for b in j2:
        for c in j3:
            for d in j4:
                for e in j5:
                    for f in j6:
                        theta = [a, b, c, d, e, f, 0]
                        T     = np.zeros((num_j+1,4,4))
                        T[0]  = np.eye(4)
                        for i in range(1,num_j+1):
                            alpha = (-1)**i*math.pi/2 if i < num_j else 0
                            T[i]  = calc_DH( theta[i-1], d_list[i-1], 0, alpha)
    
                        Transf[0] = T[0]          
                        for i in range(1,num_j+1):
                            Transf[i] = Transf[i-1]@T[i]
            
                        end_posx = np.append(end_posx,Transf[7,0,3])
                        end_posy = np.append(end_posy,Transf[7,1,3])
                        end_posz = np.append(end_posz,Transf[7,2,3])
                        count_step += 1
print("Steps",count_step)

# In[18]:

get_ipython().run_line_magic('matplotlib', 'notebook')

# In[19]:
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
from numpy.random import randn
from scipy import array, newaxis


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
