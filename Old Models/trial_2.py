import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces

class QuadrupedEnv(gym.Env):
    def __init__(self):
        super(QuadrupedEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        # Extended observation space to include orientation
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)

        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.terrain = None
        self.robot = None
        self.target_position = [100, 0, 0]  # Example target position

    def reset(self):
        # Reset the environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Load terrain
        self.load_terrain()

        # Load quadruped
        self.robot = p.loadURDF("quadruped/quadruped.urdf", basePosition=[0, 0, 0.3], baseOrientation=p.getQuaternionFromEuler([0, np.pi, 0]))

        # Let the physics settle
        for _ in range(10):
            p.stepSimulation()

        # Initialize observation
        observation = self.get_observation()
        return observation

    def load_terrain(self):
        # Define terrain type
        terrain_type = 'plane'  # Change as needed: 'plane', 'hilly', 'stairs', 'random_uneven'

        if terrain_type == 'plane':
            self.terrain = p.loadURDF("plane.urdf")
        elif terrain_type == 'hilly':
            # Example of hilly terrain
            # Similar implementation as previous example
            pass
        elif terrain_type == 'stairs':
            # Example of stairs terrain
            # Similar implementation as previous example
            pass
        elif terrain_type == 'random_uneven':
            # Example of random uneven terrain
            # Similar implementation as previous example
            pass

    def step(self, action):
        # Apply action to the robot
        p.setJointMotorControlArray(bodyUniqueId=self.robot, 
                                    jointIndices=range(12), 
                                    controlMode=p.POSITION_CONTROL, 
                                    targetPositions=action)

        # Simulate one timestep forward
        p.stepSimulation()

        # Update the observation and calculate robot's orientation
        new_observation = self.get_observation()
        _, orientation_q = p.getBasePositionAndOrientation(self.robot)
        orientation_euler = p.getEulerFromQuaternion(orientation_q)

        # Calculate the reward and done based on robot position and orientation
        robot_position = p.getBasePositionAndOrientation(self.robot)[0]
        distance_to_target = np.linalg.norm(np.array(robot_position) - np.array(self.target_position))
        stability_factor = -abs(orientation_euler[1])  # Penalize tilt along one axis
        reward = -distance_to_target + stability_factor  # Combine distance and stability in reward
        done = distance_to_target < 0.5 or abs(orientation_euler[1]) > 0.5  # Done if close to target or too tilted

        return new_observation, reward, done, {}

    def get_observation(self):
        # Return an observation that includes base position and orientation
        position, orientation = p.getBasePositionAndOrientation(self.robot)
        orientation_euler = p.getEulerFromQuaternion(orientation)
        return np.array(list(position) + list(orientation_euler))

    def render(self, mode='human'):
        # Optional rendering logic
        pass

    def close(self):
        p.disconnect()

# Example usage of the environment
env = QuadrupedEnv()
observation = env.reset()

for _ in range(50000):  # Run for a limited number of steps for testing
    action = env.action_space.sample()  # Random action
    observation, reward, done, info = env.step(action)
    if done:
        break

env.close()
