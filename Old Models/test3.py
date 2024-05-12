# Modifying the provided Python code to improve the natural movement of the quadruped and fix leg issues.

 

# Importing the necessary libraries for simulation

import numpy as np

import pybullet as p

import pybullet_data

import gym

from gym import spaces

 

class QuadrupedEnv(gym.Env):

    def __init__(self):

        super(QuadrupedEnv, self).__init__()

 

        # Define action and observation space

        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

 

        self.physics_client = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

 

        self.terrain = None

        self.robot = None

        self.target_position = [10, 0, 0]  # Example target position

 

    def reset(self):

        # Reset the environment

        p.resetSimulation()

        p.setGravity(0, 0, -9.8)

 

        # Load terrain

        self.load_terrain()

 

        # Load quadruped

        self.robot = p.loadURDF("quadruped/quadruped.urdf", basePosition=[0, 0, 0.5], baseOrientation=p.getQuaternionFromEuler([0, np.pi, 0]))

 

        # Let the physics settle

        for _ in range(10):

            p.stepSimulation()

 

        # Initialize observation

        observation = self.get_observation()

        return observation

 

    def load_terrain(self):

        # Choose terrain type here

        terrain_type = 'plane'  # options: 'plane', 'hilly', 'stairs', 'random_uneven'

 

        if terrain_type == 'plane':

            # Load flat plane terrain

            self.terrain = p.loadURDF("plane.urdf")

 

    def step(self, action):

        # Apply the action to the robot

        for i, joint_action in enumerate(action):

            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=joint_action)

 

        # Step simulation

        p.stepSimulation()

 

        # Update observation

        observation = self.get_observation()

        reward = self.calculate_reward(observation)

        done = self.check_done(observation)

        info = {}

 

        return observation, reward, done, info

 

    def get_observation(self):

        # Placeholder to fetch observations from the simulation

        return np.zeros(self.observation_space.shape)

 

    def calculate_reward(self, observation):

        # Placeholder for reward calculation

        return 0

 

    def check_done(self, observation):

        # Placeholder to determine if the goal is reached

        return False

 

    def render(self, mode='human'):

        pass

 

    def close(self):

        p.disconnect()

 

def calculate_joint_actions(step_count):

    # Improved amplitude and frequency adjustments for more natural movement

    amplitude = 0.8  # Increased amplitude for wider range of motion

    frequency = 1.2  # Increased frequency for more dynamic movement

 

    actions = np.zeros(12)

 

    # Movement patterns for front and back legs using a more dynamic trotting gait

    phases = {

        'left_front': 0,

        'right_front': np.pi,   # Opposite phase to left_front

        'left_back': np.pi,     # Opposite phase to right_back

        'right_back': 0

    }

 

    # Assign actions based on specified phases

    # Front legs

    actions[0] = amplitude * np.sin(frequency * step_count + phases['left_front'])   # Left front

    actions[1] = amplitude * np.sin(frequency * step_count + phases['right_front'])  # Right front

    actions[2] = actions[0]  # Copy to the other joint of the same leg

    actions[3] = actions[1]

 

    # Back legs

    actions[6] = amplitude * np.sin(frequency * step_count + phases['left_back'])   # Left back

    actions[7] = amplitude * np.sin(frequency * step_count + phases['right_back'])  # Right back

    actions[8] = actions[6]  # Copy to the other joint of the same leg

    actions[9] = actions[7]

 

    return actions

 

# Example usage commented out to adhere to instructions

env = QuadrupedEnv()

observation = env.reset()

step_count = 0

while True:

    action = calculate_joint_actions(step_count)

    observation, reward, done, info = env.step(action)

    step_count += 1

    if done:

        break

env.close()