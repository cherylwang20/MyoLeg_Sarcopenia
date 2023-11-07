# MyoLeg_Sarcopenia
MyoLeg is the MyoSuite model of the lower limb.

**Getting started**
The following things are required:

Python 3.7.1 pip install python
MyoSuitepip install -U myosuite
Mujoco pip install mujoco
Gym pip install gym
mj_envs $ git clone --recursive https://github.com/vikashplus/mj_envs.git; cd mj_envs and then $ pip install -e .
stable_baselines3 pip install stable-baselines3 
After you have installed this it should let you go into the tasks.

**Tasks**
Standing Balance, Standing Balance with Perturbation, Standing Balance with Leg Sarcopenia
The goal is to keep the model standing still, without falling.

How to use:

Train the model python MyoLeg_script_test/standingBalance/train_leg_local.py 
See results python MyoLeg_script_test/standingBalance/Eval_leg.py 

More to be developed.
