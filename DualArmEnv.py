# In[1]:
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.InteractiveSession()

# In[2]:

#####################  Global variables  ####################
global eps,Ez,num_j,d_time
eps    = 10**-6
Ez     = np.array([0,0,1])
num_j  = 6   # [joints]
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
std_dev    = 0.3 # OU_noise


# In[3]:


# A skew-symmetric operator
def tilde(x): 
    w = np.array([
        [    0, -x[2],  x[1]],
        [ x[2],     0, -x[0]],
        [-x[1],  x[0],    0]])
    return w


# In[24]:


# Euler angle rotation 
def Rx(theta):
    return np.matrix([[  1,    0           ,    0           ],
                       [ 0, math.cos(theta),-math.sin(theta)],
                       [ 0, math.sin(theta), math.cos(theta)]])
  
def Ry(theta):
    return np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                       [    0           , 1,    0           ],
                       [-math.sin(theta), 0, math.cos(theta)]])
  
def Rz(theta):
    return np.matrix([[  math.cos(theta), -math.sin(theta), 0 ],
                       [ math.sin(theta), math.cos(theta) , 0 ],
                       [    0           ,    0            , 1 ]])
# In[25]:
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

# In[26]:

def calc_DH(theta, d, a, alpha):
    return TRz(theta)@Tz(d)@Tx(a)@TRx(alpha)

# In[27]:

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

# In[28]:

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

# In[29]:

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

# In[31]:

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


# In[51]:


class ArmEnv(object):     
    # Class variables - None   
    def __init__(self, MAX_EPISODES=5000, MAX_EP_STEPS=300):    
        # Instance variables
        self.MAX_EPISODES    = MAX_EPISODES
        self.MAX_EP_STEPS    = MAX_EP_STEPS
        self.Time            = [i*d_time for i in range(self.MAX_EP_STEPS)]   
        self.action_dim      = 2*num_j          # Joints angular vel[rad/s]
        self.done            = False
        self.on_goal         = 0
        self.p_end1_print    = np.zeros((MAX_EP_STEPS, 3))
        self.p_end2_print    = np.zeros((MAX_EP_STEPS, 3))
        self.p_tar1_print    = np.zeros((MAX_EP_STEPS, 3))
        self.p_tar2_print    = np.zeros((MAX_EP_STEPS, 3))
        self.dist1_print     = np.zeros(MAX_EP_STEPS)
        self.dist2_print     = np.zeros(MAX_EP_STEPS)
        self.angl_1_1_print  = np.zeros(MAX_EP_STEPS) 
        self.angl_1_2_print  = np.zeros(MAX_EP_STEPS)
        self.angl_1_3_print  = np.zeros(MAX_EP_STEPS)
        self.angl_1_4_print  = np.zeros(MAX_EP_STEPS)
        self.angl_1_5_print  = np.zeros(MAX_EP_STEPS)
        self.angl_1_6_print  = np.zeros(MAX_EP_STEPS)
        self.angl_2_1_print  = np.zeros(MAX_EP_STEPS)
        self.angl_2_2_print  = np.zeros(MAX_EP_STEPS)
        self.angl_2_3_print  = np.zeros(MAX_EP_STEPS)
        self.angl_2_4_print  = np.zeros(MAX_EP_STEPS)
        self.angl_2_5_print  = np.zeros(MAX_EP_STEPS)
        self.angl_2_6_print  = np.zeros(MAX_EP_STEPS)
        Time                 = d_time*np.array(range(MAX_EP_STEPS))
        
        self.joint_bound      = np.zeros(num_j+1, dtype = [('L',np.float32),('U',np.float32)])
        self.joint_bound['L'] = [0,-20, 90,-100, 90,-80,-80]
        self.joint_bound['U'] = [0, 20,180, 100,180, 80, 80]

        self.action_bound     = [ -0.1, 0.1 ]
        # 0   : Base 
        # 1~6 : Joint
        self.body_prop_info = np.zeros(2*num_j+1, dtype=[('mass', np.float32),('link', np.float32),('Ixx', np.float32),('Iyy',np.float32),('Izz',np.float32)])
        self.body_prop_info['mass']  = [9.157, 0.171, 0.25, 0.433, 0.25, 0.433, 0.029, 0.171, 0.25, 0.433, 0.25, 0.433, 0.029]
        self.body_prop_info['link']  = [    1,  0.16, 0.06,  0.13, 0.06,  0.13, 0.04 ,  0.16, 0.06,  0.13, 0.06,  0.13, 0.04 ]      
        # base -> 좌측 : X 17.3 mm, Y -44.337 mm, Z 54.2 mm    우측 : X -17.3 mm, Y -44.337 mm, Z 54.2 mm
        self.body_prop_info['Ixx']   = [  0.177, 116.3*eps, 74.26*eps, 688.6*eps, 125.2*eps, 403.1*eps, 6.094*eps, 116.3*eps, 74.26*eps, 688.6*eps, 125.2*eps, 403.1*eps, 6.094*eps]
        self.body_prop_info['Iyy']   = [  0.129, 460.3*eps, 104.8*eps, 593.1*eps, 106.7*eps, 659.7*eps, 6.654*eps, 460.3*eps, 104.8*eps, 593.1*eps, 106.7*eps, 659.7*eps, 6.654*eps]
        self.body_prop_info['Izz']   = [  0.13,  491.9*eps, 121.9*eps, 160.1*eps, 73.82*eps,   379*eps, 6.378*eps, 491.9*eps, 121.9*eps, 160.1*eps, 73.82*eps,   379*eps, 6.378*eps]

        self.Mass   = sum(self.body_prop_info['mass'])            # Arm + Base mass

        # 0 ~ 6 : Joint -> p, z
        # 0     : Base  -> r0  
        # 1 ~ 6 : Link  -> r, v, w
        self.Arm_kin_info = np.zeros( 2*num_j+1, dtype=[('theta', np.float32)]) 
        self.State_info   = np.zeros((2*num_j+1,3),dtype=[('p', np.float32),('r', np.float32),('z', np.float32),('v', np.float32),('w', np.float32)])
        self.Vector_info  = np.zeros(3,dtype=[('rg', np.float32)])

        # 0 ~ 6 : Joints (0 ~ 6)
        # -1    : Base
        self.AA = np.zeros((2*num_j+1,3,3), dtype=[('A',np.float32)])     # Rotation matrices
        self.AA['A'][0]      = np.eye(3)
        self.AA['A'][-1]     = np.eye(3)
        self.Transf = np.zeros((2*num_j+1,4,4), dtype=[('T',np.float32)]) # Transformation matrices
        self.Transf['T'][0]  = np.eye(4)
        self.Transf['T'][-1] = np.eye(4)

        # 0 ~ 5 : Joints (1 ~ 6)
        self.J_info      = np.zeros((2*num_j,3,2*num_j),dtype=[('Jp', np.float32),('JH',np.float32)])
        self.Jp_info     = np.zeros((3,2*num_j),dtype=[('JpPhi', np.float32),('Pphi', np.float32)])

        # Target information ( Starlink )
        self.Target_info = np.zeros(3, dtype=[('r', np.float32),('h_w_radius', np.float32)]) # Workspace needs to be specified
        self.Target_info['r']          = [ 0,    0, 1.1]
        self.Target_info['h_w_radius'] = [ 1, 0.01, 0.4]

        self.beta   = math.atan2(self.Target_info['h_w_radius'][2],self.Target_info['h_w_radius'][0])

        # DDPG variables
        self.state_dim = 64
        
        # Visualization variables
        self.count_step = 0    
        
    def step(self, action):
        self.count_step   += 1
        self.done          = False
        action             = np.clip(action, *self.action_bound) 
        
        # Transformation
        # DH parameter       
        T    = np.zeros((num_j+1,4,4))
        T[0] = np.eye(4)
        Base_A = self.AA['A'][-1]
        T[0,:3,:3] = self.AA['A'][-1]
        T[0,:3, 3] = self.AA['A'][-1]@np.array([ 0.0173, -0.044337, 0.0542]) + self.State_info['r'][0]
        theta = self.Arm_kin_info['theta'][1:num_j+1]
        link = self.body_prop_info['link']
        T[1]  = calc_DH( theta[0], link[0], 0, 0)
        T[2]  = calc_DH( 0, link[1], 0, theta[1])
        T[3]  = calc_DH( theta[2], -link[2], 0, 0)
        T[4]  = calc_DH( 0, -link[3], 0, theta[3])
        T[5]  = calc_DH( theta[4], link[4], 0, 0)
        T[6]  = calc_DH( theta[5], link[5], 0, 0)
                            
        self.Transf['T'][0] = T[0]           
        for i in range(1,num_j+1):
            self.Transf['T'][i] = self.Transf['T'][i-1]@T[i]
                            
        self.State_info['p'][:num_j+1] = self.Transf['T'][:num_j+1,:3, 3]
        self.State_info['z'][:num_j+1] = self.Transf['T'][:num_j+1,:3, 2]
        self.AA['A'][:num_j+1]         = self.Transf['T'][:num_j+1,:3,:3]
        T    = np.zeros((num_j+1,4,4))
        T[0] = np.eye(4)
        T[0,:3,:3] = self.AA['A'][-1]
        T[0,:3, 3] = self.AA['A'][-1]@np.array([-0.0173,-0.044337, 0.0542]) + self.State_info['r'][0]
        theta = self.Arm_kin_info['theta'][num_j+1:2*num_j+1]
        T[1]  = calc_DH( theta[0], -link[0], 0, 0)
        T[2]  = calc_DH( 0, -link[1], 0, theta[1])
        T[3]  = calc_DH( theta[2], -link[2], 0, 0)
        T[4]  = calc_DH( 0, -link[3], 0, theta[3])
        T[5]  = calc_DH( theta[4], -link[4], 0, 0)
        T[6]  = calc_DH( theta[5], -link[5], 0, 0)
                                    
        for i in range(1,num_j+1):
            self.Transf['T'][i+num_j] = self.Transf['T'][i-1]@T[i]
        
        self.State_info['p'][num_j+1:] = self.Transf['T'][num_j+1:2*num_j+1,:3,3]
        self.State_info['z'][num_j+1:] = self.Transf['T'][num_j+1:,:3,2]
        self.AA['A'][num_j+1:]         = self.Transf['T'][num_j+1:,:3,:3]
        self.AA['A'][-1] = Base_A
        
        X = self.State_info['p']
        for j in range(1,2*num_j):
            self.State_info['r'][j] = (X[j+1] + X[j])/2
       
        # GJM
        # 1. Calculate the inertia matrix  
        self.Vector_info['rg'] = self.body_prop_info['mass']@self.State_info['r']/self.Mass
        for i in range(1,num_j+1):
            # Arm 1
            X = np.zeros((2*num_j,3))
            for j in range(1,i+1):
                X[j-1][:] = np.array([np.cross(self.State_info['z'][i], self.State_info['r'][i] - self.State_info['p'][j])])
            self.J_info['Jp'][i-1] = X.T     
            # Arm 2
            X = np.zeros((2*num_j,3))
            for j in range(1,i+1):
                X[j-1+num_j][:] = np.array([np.cross(self.State_info['z'][i], self.State_info['r'][num_j+i] - self.State_info['p'][num_j+j])])
            self.J_info['Jp'][i-1+num_j] = X.T 
                        
        self.Jp_info['JpPhi'] = np.zeros((3,2*num_j))
        for i in range(1,num_j+1):
            self.Jp_info['JpPhi'] += self.body_prop_info['mass'][i]*np.sum(self.J_info['Jp'][:i], axis=0)
            self.Jp_info['JpPhi'] += self.body_prop_info['mass'][i]*np.sum(self.J_info['Jp'][num_j:num_j+i], axis=0)
               
        X = np.zeros((2*num_j,3))
        for i in range(num_j):
            X[i][:] = self.State_info['z'][i+1]
            self.J_info['JH'][i] = X.T
            X[i+num_j][:] = self.State_info['z'][i+1+num_j]
            self.J_info['JH'][i+num_j] = X.T
            
        self.Jp_info['Pphi'] = np.zeros((3,2*num_j))
        for i in range(1,num_j+1):
            I_matrix = np.zeros((3,3))
            I_matrix[0][0] = self.body_prop_info['Ixx'][i]
            I_matrix[1][1] = self.body_prop_info['Iyy'][i]
            I_matrix[2][2] = self.body_prop_info['Izz'][i]
            self.Jp_info['Pphi'] += I_matrix@self.J_info['JH'][i-1] + self.body_prop_info['mass'][i]*tilde(self.State_info['r'][i])@self.J_info['Jp'][i-1]
            self.Jp_info['Pphi'] += I_matrix@self.J_info['JH'][num_j+i-1] + self.body_prop_info['mass'][i]*tilde(self.State_info['r'][num_j+i])@self.J_info['Jp'][num_j+i-1]
        
        Pw = np.zeros((3,3))
        Pw[0][0] = sum(self.body_prop_info['Ixx'])
        Pw[1][1] = sum(self.body_prop_info['Iyy'])
        Pw[2][2] = sum(self.body_prop_info['Izz'])
        
        for i in range(1,num_j+1):
            Pw -= self.body_prop_info['mass'][i]*tilde(self.State_info['r'][i])@tilde(self.State_info['r'][i] - self.State_info['r'][0])    
            Pw -= self.body_prop_info['mass'][i]*tilde(self.State_info['r'][i+num_j])@tilde(self.State_info['r'][i+num_j] - self.State_info['r'][0])
        
        X    = self.Mass*np.concatenate((np.eye(3), tilde(self.Vector_info['rg'])),axis = 0)     # 세로 합치기
        Y    = np.concatenate((-self.Mass*tilde(self.Vector_info['rg'] - self.State_info['r'][0]),Pw),axis = 0)  # 세로 합치기
        Hs   = np.concatenate((X,Y),axis = 1)                                              # 가로 합치기 
        Hm   = np.concatenate((self.Jp_info['JpPhi'],self.Jp_info['Pphi']),axis = 0)       # 세로 합치기
        E12  = -LA.inv(Hs)@Hm
        
        p0ex1 = tilde(self.State_info['p'][-1-num_j] - self.State_info['r'][0])
        X     = np.concatenate((np.eye(3),np.zeros((3,3))), axis = 0)  # 세로 합치기
        Y     = np.concatenate((-p0ex1,np.eye(3)),axis = 0)            # 세로 합치기        
        J0_u  = np.concatenate((X,Y),axis = 1)                         # 가로 합치기
        
        p0ex2 = tilde(self.State_info['p'][-1] - self.State_info['r'][0]) 
        Y     = np.concatenate((-p0ex2,np.eye(3)),axis = 0)            # 세로 합치기        
        J0_l  = np.concatenate((X,Y),axis = 1)                         # 가로 합치기
        J0    = np.concatenate((J0_u,J0_l),axis = 0)                   # 세로 합치기   
          
        X  = np.concatenate((self.J_info['Jp'][-1-num_j],self.J_info['JH'][-1-num_j]),axis = 0) # 세로 합치기
        Y  = np.concatenate((self.J_info['Jp'][-1],self.J_info['JH'][-1]),axis = 0)             # 세로 합치기
        Jm = np.concatenate((X,Y),axis=0)
        
        GJ  = Jm + J0@E12 # 12,12 + 12,12
        
        GJM = np.concatenate((E12,GJ),axis = 0)                  # 세로 합치기 
        v_w = GJM@action   
        
        ## State update  
        # Flight dynamics
        r_temp = self.State_info['r']
        v_temp = self.State_info['v']
        w_temp = self.State_info['w']
        w0     = w_temp[0]
        
        self.State_info['r'][0] += v_temp[0]*d_time
        for i in range(1,num_j):
            self.State_info['v'][i] = v_temp[0] + np.cross( w_temp[0], r_temp[i] - r_temp[0] ) + np.sum(np.cross(self.State_info['z'][:i],r_temp[i] - self.State_info['p'][:i]),axis=0)  
        for i in range(num_j+1,2*num_j):
            self.State_info['v'][i] = v_temp[0] + np.cross( w_temp[0], r_temp[i] - r_temp[0] ) + np.sum(np.cross(self.State_info['z'][:i],r_temp[i] - self.State_info['p'][:i]),axis=0)  
        
        self.State_info['v'][0]        = v_w[:3]
        self.State_info['w'][0]        = v_w[3:6]
        self.State_info['v'][-1-num_j] = v_w[6:9]
        self.State_info['w'][-1-num_j] = v_w[9:12]
        self.State_info['v'][-1]       = v_w[12:15]
        self.State_info['w'][-1]       = v_w[15:18]
        
        # Arm 1, 2 Joint 1 ~ 6 angular vel 구하기
        Norm_w = np.zeros(2*num_j)
        self.State_info['w'][1:] = np.zeros((2*num_j,3))
        for i in range(1,num_j):
            for j in range(1,i+1):
                self.State_info['w'][i] += self.State_info['z'][j]*action[j-1]
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
        self.Target_info['r'] = np.array([ 0, 0, 1.1])
        X     = self.Target_info['h_w_radius']
        alpha = (X[1]*d_time*self.count_step)%(2*math.pi)
        self.Target_info['r'][0] += X[-1]*math.sin(alpha)
        self.Target_info['r'][1] -= X[-1]*math.cos(alpha)
        
        X = np.array([math.cos(alpha/2)])
        Y = Ez*math.sin(alpha/2)
        q = np.concatenate((X,Y), axis = 0)
        
        target1_A = qtn2R(q)@Rx(self.beta)
        target2_A = target1_A
        #######################################
        X = self.Target_info['r']
        target1 = np.array([X[0]-0.01,X[1]+0.01,X[2]])
        target2 = np.array([X[0]+0.01,X[1]+0.01,X[2]])
        #######################################   
        # Capturing Distance
        dist1 = LA.norm(target1 - self.State_info['p'][-1-num_j])
        dist2 = LA.norm(target2 - self.State_info['p'][-1])
                
        # Capture Angles
        target1_y = np.zeros(3)
        for i in range(3):
            target1_y[i] = target1_A[i,1]
        endeff1_y = self.AA['A'][6,:,1]
        
        innerAB  = target1_y@endeff1_y
        AB       = LA.norm(target1_y)*LA.norm(endeff1_y)
        angle1_e  = np.arccos(innerAB/AB) # [rad]
        
        target2_y = np.zeros(3)
        for i in range(3):
            target2_y[i] = target2_A[i,1]
        endeff2_y = self.AA['A'][12,:,1]
        
        innerAB  = target2_y@endeff2_y
        AB       = LA.norm(target2_y)*LA.norm(endeff2_y)
        angle2_e  = np.arccos(innerAB/AB) # [rad]
        
        r = 10*np.exp(-40*dist1) + 10*np.exp(-40*dist2) #- 0.2*math.log10(angle1_e+eps) - 0.2*math.log10(angle2_e + eps)
        if dist1 < 0.05 or dist2 < 0.05:
            r += w_done
        #if angle1_e < 0.1 or angle2_e < 0.1:
            #r += w_done
        if dist1 < 0.04 and dist2 < 0.04 : #and angle1_e < 0.1 and angle2_e < 0.1:    
            self.done = True            
        # State
        s = np.append(self.State_info['r'][0], self.AA['A'][-1].reshape(1,-1))
        s = np.append(s,self.State_info['v'][0])
        s = np.append(s,self.Arm_kin_info['theta'])
        s = np.append(s,LA.norm(self.State_info['w'][0]))
        s = np.append(s,Norm_w)
        s = np.append(s,self.State_info['p'][-1-num_j])
        s = np.append(s,self.State_info['p'][-1])
        s = np.append(s,self.State_info['v'][-1-num_j])
        s = np.append(s,self.State_info['v'][-1])
        s = np.append(s,target1)
        s = np.append(s,target2)
        s = np.append(s,self.Target_info['h_w_radius'][1]*self.Target_info['h_w_radius'][2])
        s = np.append(s,dist1)
        s = np.append(s,dist2)
        s = np.append(s,angle1_e)
        s = np.append(s,angle2_e)
        
        # Visualization
        self.p_end1_print[self.count_step-1]  = self.State_info['p'][-1-num_j]
        self.p_end2_print[self.count_step-1]  = self.State_info['p'][-1]
        self.p_tar1_print[self.count_step-1] = target1
        self.p_tar2_print[self.count_step-1] = target2
        self.dist1_print[self.count_step-1]  = dist1
        self.dist2_print[self.count_step-1]  = dist2
        self.angl1_print[self.count_step-1]  = angle1_e          # [rad]
        self.angl2_print[self.count_step-1]  = angle2_e          # [rad]
        self.angl_1_1_print[self.count_step-1] = self.Arm_kin_info['theta'][1]
        self.angl_1_2_print[self.count_step-1] = self.Arm_kin_info['theta'][2]
        self.angl_1_3_print[self.count_step-1] = self.Arm_kin_info['theta'][3]
        self.angl_1_4_print[self.count_step-1] = self.Arm_kin_info['theta'][4]
        self.angl_1_5_print[self.count_step-1] = self.Arm_kin_info['theta'][5]
        self.angl_1_6_print[self.count_step-1] = self.Arm_kin_info['theta'][6]
        self.angl_2_1_print[self.count_step-1] = self.Arm_kin_info['theta'][num_j+1]
        self.angl_2_2_print[self.count_step-1] = self.Arm_kin_info['theta'][num_j+2]
        self.angl_2_3_print[self.count_step-1] = self.Arm_kin_info['theta'][num_j+3]
        self.angl_2_4_print[self.count_step-1] = self.Arm_kin_info['theta'][num_j+4]
        self.angl_2_5_print[self.count_step-1] = self.Arm_kin_info['theta'][num_j+5]
        self.angl_2_6_print[self.count_step-1] = self.Arm_kin_info['theta'][num_j+6]
        return s, r, self.done
    
    def reset(self):
        self.Arm_kin_info['theta'] = [ 0, 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random(), 0.1*random.random()]
        self.State_info = np.zeros((2*num_j+1,3),dtype=[('p', np.float32),('r', np.float32),('z', np.float32),('v', np.float32),('w', np.float32)])
        self.AA['A'][-1]        = np.eye(3)
        self.Transf['T'][-1]    = np.eye(4)
                    
        # Transformation
        # DH - parameter       
        T    = np.zeros((num_j+1,4,4))
        T[0] = np.eye(4)
        T[0,:3,:3] = self.AA['A'][-1]
        T[0,:3, 3] = self.AA['A'][-1]@np.array([ 0.0173, -0.044337, 0.0542]) + self.State_info['r'][0]
        theta = self.Arm_kin_info['theta'][1:num_j+1]
        link  = self.body_prop_info['link']
        T[1]  = calc_DH( theta[0], link[0], 0, 0)
        T[2]  = calc_DH( 0, link[1], 0, theta[1])
        T[3]  = calc_DH( theta[2], -link[2], 0, 0)
        T[4]  = calc_DH( 0, -link[3], 0, theta[3])
        T[5]  = calc_DH( theta[4], link[4], 0, 0)
        T[6]  = calc_DH( theta[5], link[5], 0, 0)
                            
        self.Transf['T'][0] = T[0]           
        for i in range(1,num_j+1):
            self.Transf['T'][i] = self.Transf['T'][i-1]@T[i]
                            
        self.State_info['p'][:num_j+1] = self.Transf['T'][:num_j+1,:3, 3]
        self.State_info['z'][:num_j+1] = self.Transf['T'][:num_j+1,:3, 2]
        self.AA['A'][:num_j+1]         = self.Transf['T'][:num_j+1,:3,:3]
        T    = np.zeros((num_j+1,4,4))
        T[0] = np.eye(4)
        T[0,:3,:3] = self.AA['A'][-1]
        T[0,:3, 3] = self.AA['A'][-1]@np.array([-0.0173,-0.044337, 0.0542]) + self.State_info['r'][0]
        theta = self.Arm_kin_info['theta'][num_j+1:2*num_j+1]
        T[1]  = calc_DH( theta[0], -link[0], 0, 0)
        T[2]  = calc_DH( 0, -link[1], 0, theta[1])
        T[3]  = calc_DH( theta[2], -link[2], 0, 0)
        T[4]  = calc_DH( 0, -link[3], 0, theta[3])
        T[5]  = calc_DH( theta[4], -link[4], 0, 0)
        T[6]  = calc_DH( theta[5], -link[5], 0, 0)
                                    
        for i in range(1,num_j+1):
            self.Transf['T'][i+num_j] = self.Transf['T'][i-1]@T[i]
        
        self.State_info['p'][num_j+1:] = self.Transf['T'][num_j+1:2*num_j+1,:3,3]
        self.State_info['z'][num_j+1:] = self.Transf['T'][num_j+1:,:3,2]
        self.AA['A'][num_j+1:]         = self.Transf['T'][num_j+1:,:3,:3]
        self.AA['A'][-1]        = np.eye(3)
        
        X = self.State_info['p']
        for j in range(1,2*num_j):
            self.State_info['r'][j] = (X[j+1] + X[j])/2
            
        # GJM
        # 1. Calculate the inertia matrix  
        self.Vector_info['rg'] = self.body_prop_info['mass']@self.State_info['r']/self.Mass
        for i in range(1,num_j+1):
            # Arm 1
            X = np.zeros((2*num_j,3))
            for j in range(1,i+1):
                X[j-1][:] = np.array([np.cross(self.State_info['z'][i], self.State_info['r'][i] - self.State_info['p'][j])])
            self.J_info['Jp'][i-1] = X.T     
            # Arm 2
            X = np.zeros((2*num_j,3))
            for j in range(1,i+1):
                X[j-1+num_j][:] = np.array([np.cross(self.State_info['z'][i], self.State_info['r'][num_j+i] - self.State_info['p'][num_j+j])])
            self.J_info['Jp'][i-1+num_j] = X.T     
                
        self.Jp_info['JpPhi'] = np.zeros((3,2*num_j))
        v_w = np.zeros(18) 
                   
        ## State update ##                
        # Update target rotation
        self.Target_info['r'] = np.array([ 0, 0, 1.1])
        X     = self.Target_info['h_w_radius']
        alpha = 0
    
        self.Target_info['r'][0] += X[-1]*math.sin(alpha)
        self.Target_info['r'][1] -= X[-1]*math.cos(alpha)
        
        X = np.array([math.cos(alpha/2)])
        Y = Ez*math.sin(alpha/2)
        q = np.concatenate((X,Y), axis = 0)
        
        target1_A = qtn2R(q)@Rx(self.beta)
        target2_A = target1_A
        #######################################
        X = self.Target_info['r']
        target1 = np.array([X[0]-0.01,X[1]+0.01,X[2]])
        target2 = np.array([X[0]+0.01,X[1]+0.01,X[2]])
        #######################################   
        # Capturing Distance
        dist1 = LA.norm(target1 - self.State_info['p'][-1-num_j])
        dist2 = LA.norm(target2 - self.State_info['p'][-1])
                
        # Capture Angles
        target1_y = np.zeros(3)
        for i in range(3):
            target1_y[i] = target1_A[i,1]
        endeff1_y = self.AA['A'][6,:,1]
        
        innerAB  = target1_y@endeff1_y
        AB       = LA.norm(target1_y)*LA.norm(endeff1_y)
        angle1_e  = np.arccos(innerAB/AB) # [rad]
        
        target2_y = np.zeros(3)
        for i in range(3):
            target2_y[i] = target2_A[i,1]
        endeff2_y = self.AA['A'][12,:,1]
                
        innerAB  = target2_y@endeff2_y
        AB       = LA.norm(target2_y)*LA.norm(endeff2_y)
        angle2_e = np.arccos(innerAB/AB) # [rad]
        
        X = self.Target_info['r']
        target1 = np.array([X[0]-0.01,X[1]+0.01,0])
        target2 = np.array([X[0]+0.01,X[1]+0.01,0])     
                        
        # Capturing Distance
        dist1 = LA.norm(target1 - self.State_info['p'][-1-num_j])
        dist2 = LA.norm(target2 - self.State_info['p'][-1])
              
        # State
        s = np.append(self.State_info['r'][0], self.AA['A'][-1].reshape(1,-1))
        s = np.append(s,self.State_info['v'][0])
        s = np.append(s,self.Arm_kin_info['theta'])
        s = np.append(s,np.zeros(2*num_j+1))
        s = np.append(s,self.State_info['p'][-1-num_j])
        s = np.append(s,self.State_info['p'][-1])
        s = np.append(s,self.State_info['v'][-1-num_j])
        s = np.append(s,self.State_info['v'][-1])
        s = np.append(s,target1)
        s = np.append(s,target2)
        s = np.append(s,self.Target_info['h_w_radius'][1]*self.Target_info['h_w_radius'][2])
        s = np.append(s,dist1)
        s = np.append(s,dist2)
        s = np.append(s,angle1_e)
        s = np.append(s,angle2_e)
        
        #################
        # Visualization #
        self.count_step   = 0
        self.on_goal      = 0
        self.p_end1_print = np.zeros((MAX_EP_STEPS, 3))
        self.p_end2_print = np.zeros((MAX_EP_STEPS, 3))
        self.p_tar1_print = np.zeros((MAX_EP_STEPS, 3))
        self.p_tar2_print = np.zeros((MAX_EP_STEPS, 3))
        self.dist1_print  = np.zeros(MAX_EP_STEPS)
        self.dist2_print  = np.zeros(MAX_EP_STEPS)
        self.angl1_print  = np.zeros(MAX_EP_STEPS)  # [rad]
        self.angl2_print  = np.zeros(MAX_EP_STEPS)  # [rad]
        self.angl_1_1_print  = np.zeros(MAX_EP_STEPS) 
        self.angl_1_2_print  = np.zeros(MAX_EP_STEPS)
        self.angl_1_3_print  = np.zeros(MAX_EP_STEPS)
        self.angl_1_4_print  = np.zeros(MAX_EP_STEPS)
        self.angl_1_5_print  = np.zeros(MAX_EP_STEPS)
        self.angl_1_6_print  = np.zeros(MAX_EP_STEPS)
        self.angl_2_1_print  = np.zeros(MAX_EP_STEPS)
        self.angl_2_2_print  = np.zeros(MAX_EP_STEPS)
        self.angl_2_3_print  = np.zeros(MAX_EP_STEPS)
        self.angl_2_4_print  = np.zeros(MAX_EP_STEPS)
        self.angl_2_5_print  = np.zeros(MAX_EP_STEPS)
        self.angl_2_6_print  = np.zeros(MAX_EP_STEPS)    
        return s
    
    def print_figures(self,MAX_EP_STEPS):
        fig = plt.figure(figsize=(10,10))    
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.p_end1_print[:,0], self.p_end1_print[:,1], self.p_end1_print[:,2], c= "red")
        ax.scatter(self.p_tar1_print[:,0], self.p_tar1_print[:,1], self.p_tar1_print[:,2], c="blue")
        
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(["Arm 1 Positions of End Effector(red) & Target(blue)"])      
        
        fig = plt.figure(figsize=(10,10))    
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.p_end2_print[:,0], self.p_end2_print[:,1], self.p_end2_print[:,2], c= "red")
        ax.scatter(self.p_tar2_print[:,0], self.p_tar2_print[:,1], self.p_tar2_print[:,2], c="blue")
        
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(["Arm 2 Positions of End Effector(red) & Target(blue)"])         
        
        plt.figure(figsize=(6,6))
        plt.plot(self.Time, self.dist1_print)
        plt.title('Arm 1 Distance[m] vs Time[sec]')
        
        plt.figure(figsize=(6,6))
        plt.plot(self.Time, self.dist2_print)
        plt.title('Arm 2 Distance[m] vs Time[sec]')
        
        plt.figure(figsize=(6,6))
        plt.plot(self.Time, self.angl1_print)
        plt.title('Angle 1 error[rad] vs Time[sec]')
        
        plt.figure(figsize=(6,6))
        plt.plot(self.Time, self.angl2_print)
        plt.title('Angle 2 error[rad] vs Time[sec]')
        
        # Joint angle
        plt.figure(figsize=(6,6))
        plt.plot(self.Time, self.angl_1_1_print,'r')
        plt.plot(self.Time, self.angl_1_2_print,'g')
        plt.plot(self.Time, self.angl_1_3_print,'b')
        plt.plot(self.Time, self.angl_1_4_print,'c')
        plt.plot(self.Time, self.angl_1_5_print,'m')
        plt.plot(self.Time, self.angl_1_6_print,'y')
        plt.title('Arm 1 Angle[rad] vs Time[sec]')
        
        plt.figure(figsize=(6,6))
        plt.plot(self.Time, self.angl_2_1_print,'r')
        plt.plot(self.Time, self.angl_2_2_print,'g')
        plt.plot(self.Time, self.angl_2_3_print,'b')
        plt.plot(self.Time, self.angl_2_4_print,'c')
        plt.plot(self.Time, self.angl_2_5_print,'m')
        plt.plot(self.Time, self.angl_2_6_print,'y')
        plt.title('Arm 2 Angle[rad] vs Time[sec]')
        
        plt.show()
                
    def render(self):
        pass

# In[52]:

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
MAX_EP_STEPS = 300

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
                    
                #print('Error angle: %.1f' %(s[-1]))
                distance.append(s[-2])   
                break
            ep_step += 1
        if ep_r > 200:
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
plt.figure(figsize=(6,6))
plt.plot(xaxis,np.array(reward_plt))
plt.title("Returns per Episode")

plt.figure(figsize=(6,6))
plt.plot(np.arange(len(success_rate)),np.array(success_rate))
plt.title("success rate")

plt.show()
