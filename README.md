# Space-manipulator
Motion Planning of Space Manipulator for Capturing a Tumbling Non-cooperative Target using Reinforcement Learning

These are the codes that were used for my master's thesis.
You could read the thesis by searching the name "Dahyun Lee" on the kaist library website.

About the training codes,
There are so many factors that affect training results.
Due to many trials, there are many alternative values in the code.
For example, I tried the reward functions, exp, and log to compare the performance to our system. The performance depends on the system, which means the reward function needs scaling and reshaping.
Also, the joint limitation was considered in training earlier and removed later to see the training effectiveness.
The range of random initial join configuration could be changed as well.
Other factors that affect training results; action range, std_dev of OU noise, target position, rotating speed (slower is better), MAX_EP_STEPS, etc.
