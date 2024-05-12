# from ctypes import pointer
# from click import pass_context
import pybullet as p
# import pybullet_envs
import pybullet_data
# import torch 
import gym
from gym import spaces
# import time
from stable_baselines3 import PPO 
# from stable_baselines3 import SAC 
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.env_checker import check_env
import numpy as np        
import math
import os
import inv_kine.inv_kine as ik
import matplotlib.pyplot as plt

# for plot
posx_list = []
posy_list = []
posz_list = []
velx_list = []
rot_list = []
pow_list = []

# see tensorboard : tensorboard --logdir=log (open terminal in final_project dir)
    
class TestudogEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(TestudogEnv, self).__init__()
        self.state = self.init_state()
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(45,), dtype=np.float32)
        
    def init_state(self):
        self.count = 0
        # p.connect(p.DIRECT)
        if p.isConnected():
            p.disconnect()

        # Now, connect to a new simulation
        p.connect(p.GUI)
        # p.connect(p.GUI, options="--logtostderr --logLevel=3")
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
        self.testudogid = p.loadURDF("./urdf/testudog.urdf",[0,0,0.25],[0,0,0,1])
        if self.testudogid is None or self.testudogid < 0:
            print("Failed to load testudog URDF.")
            return None  # or handle the error differently
        focus, _ = p.getBasePositionAndOrientation(self.testudogid)

        p.resetDebugVisualizerCamera(cameraDistance=1,cameraYaw=-90,cameraPitch=0,cameraTargetPosition=focus)
        
        # observation --> body: pos, rot, lin_vel ang_vel / joints: pos, vel / foot position? / foot contact?
        body_pos = p.getLinkState(self.testudogid,0)[0]
        body_rot = p.getLinkState(self.testudogid,0)[1]
        body_rot_rpy = p.getEulerFromQuaternion(body_rot) 
        body_lin_vel = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[6]
        body_ang_vel = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[7]
        joint_pos = []
        joint_vel = []
        joint_torque = []
        for i in range(12):
            joint_pos.append(p.getJointState(self.testudogid,i)[0])
            joint_vel.append(p.getJointState(self.testudogid,i)[1])        
            joint_torque.append(p.getJointState(self.testudogid,i)[3]) 
        # obs = list(body_pos) + list(body_rot_rpy)[0:2] + list(body_lin_vel) + list(body_ang_vel) + joint_pos + joint_vel + joint_torque
        obs = list(body_pos) + list(body_lin_vel) + list(body_rot_rpy) + joint_pos + joint_vel + joint_torque
        obs = np.array(obs).astype(np.float32)
        return obs
    
    def reset(self):
        p.disconnect()
        obs = self.init_state()
        self.state = obs
        return obs
        
    def step(self, action):
        # Modify the leg positioning logic here:
        # Assuming the structure of 'action' is such that:
        # action[0:3] corresponds to the right legs and action[3:6] corresponds to the left legs
        # This maps actions to the legs such that the left and right sides move in unison.
        action_legpos = np.array([
            [action[0], action[0], action[3], action[3]],  # X positions for right front, right back, left front, left back
            [action[1], action[1], action[4], action[4]],  # Y positions
            [action[2], action[2], action[5], action[5]]   # Z positions
        ])

        # Convert the global leg positions to joint angles using inverse kinematics
        joint_angle = ik.inv_kine(ik.global2local_legpos(action_legpos, x_global, y_global, z_global, roll, pitch, yaw))
        joint_angle = np.reshape(np.transpose(joint_angle), [1, 12])[0]

        # Set joint motor controls
        vel1 = action[6:9]  # Assuming these correspond to right legs' velocities
        vel2 = action[9:12] # Assuming these correspond to left legs' velocities
        p.setJointMotorControlArray(self.testudogid, list(range(12)), p.POSITION_CONTROL,
            targetPositions=joint_angle, targetVelocities=np.block([vel1, vel2]), positionGains=4*[0.02, 0.02, 0.02], velocityGains=4*[0.1, 0.1, 0.1])

        # Update the camera focus
        focus, _ = p.getBasePositionAndOrientation(self.testudogid)
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=focus)
        p.stepSimulation()

        # Update observations and calculate rewards
        body_pos = p.getLinkState(self.testudogid, 0)[0]
        body_rot = p.getLinkState(self.testudogid, 0)[1]
        body_rot_rpy = p.getEulerFromQuaternion(body_rot)
        body_lin_vel = p.getLinkState(self.testudogid, 0, computeLinkVelocity=1)[6]
        body_ang_vel = p.getLinkState(self.testudogid, 0, computeLinkVelocity=1)[7]
        joint_pos, joint_vel, joint_torque, joint_pow = [], [], [], []
        for i in range(12):
            joint_info = p.getJointState(self.testudogid, i)
            joint_pos.append(joint_info[0])
            joint_vel.append(joint_info[1])
            joint_torque.append(joint_info[3])
            joint_pow.append(joint_info[1] * joint_info[3])  # Power = torque * velocity

        obs = list(body_pos) + list(body_lin_vel) + list(body_rot_rpy) + joint_pos + joint_vel + joint_torque
        obs = np.array(obs).astype(np.float32)
        info = {}
        self.count += 1
        reward = self.compute_reward(body_pos, joint_pow, body_rot_rpy, body_lin_vel)
        done = self.is_done(body_rot_rpy, self.count)

        return obs, reward, done, info


if (__name__ == '__main__'):
    # set save directory
    model_dir ="./models/PPO"
    log_dir = "./log"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # testudog initial state
    x_global = 0
    y_global = 0
    z_global = 0.15
    roll = 0
    pitch = 0
    yaw = 0
        
    # create model
    env = TestudogEnv()
    # check_env(env)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    TIMESTEPS = 50000
    count = 1
    
    # load model
    # model_path = f"{model_dir}/15450000.zip"
    # model = PPO.load(model_path,env=env)
    # count = int(15450000/TIMESTEPS)
    
    # # train loop   
    while(True):
        print(count)
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{model_dir}/{TIMESTEPS*count}")
        count += 1
        if True == False:
            break
    
    # run trained model
    # episodes = 1
    # for ep in range(episodes):
    #     obs = env.reset()
    #     done = False
    #     while not done and env.count<2000:
    #         action, _ = model.predict(obs)
    #         obs, reward, done, info = env.step(action)
    #         time.sleep(1/240)
    
    # size = len(posx_list)
    # time_sim = np.arange(0,size,1)/240
    # fig, axes = plt.subplots(3, 2)   
    # axes[0,0].plot(time_sim, posx_list) 
    # axes[1,0].plot(time_sim, posy_list) 
    # axes[2,0].plot(time_sim, posz_list) 
    # axes[0,1].plot(time_sim, velx_list) 
    # axes[1,1].plot(time_sim, rot_list) 
    # axes[2,1].plot(time_sim, pow_list) 
    
    # plt.plot()
            
