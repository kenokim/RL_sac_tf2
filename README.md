# Soft actor critic tensorflow 2.x implementation and explanation

Soft actor critic : Haarnoja et. al. (201812)

# Requirements
tensorflow >= 2.1  
pybullet == 2.6 (alternatively, you can use openai gym mujoco)  
python >= 3.6  


# Environment
1. Pybullet  
* Assuming you are using google colab, following codes will provide the enjoyable mujoco environment.  
!pip install pybullet  
!git clone https://github.com/benelot/pybullet-gym.git  
cd pybullet-gym  
!pip install -e  
import gym  
import pybulletgym  
env = gym.make('HopperPyBulletEnv-v0')  

2. Openai gym Mujoco  
Recommended to use Linux or MacOs as your development environment. Otherwise, you'll encounter a lot of bugs! ( I did :worried: )  
* Go to the official Mujoco homepage, download an appropriate license. Then, follow steps represented on the page.
This enables having an experiment on your local pc! :satisfied:


# Algorithm explanations
I'll explain what I understood studying sac.   
* Soft actor critic is a model-free, off-policy algorithm. (Probably one of the simplest rl algorithms.)  
* It basically operates by iterating the following steps:
1. Update Q value function (the critic) : how much reward is expected as taking an action from a state.  
2. Update the policy (the actor) by matching the Kullback-Leibler divergence.   
Though it is astonishingly simple, it works great. 




# Reference  
I took and changed some part of Stanford's open course assignment codes. That is "creating sampling" part. 

First writing : 20.01.xx  
Last modification : 20.05.xx  


