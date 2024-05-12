import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import os

class QuadrupedEnv(gym.Env):
    """Custom Environment for Quadruped that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(QuadrupedEnv, self).__init__()

        # Connect to PyBullet
        self.physicsClient = p.connect(p.GUI)
        self.robot_id = self.setup_robot()
        num_joints = len(self.joint_ids)  # Retrieve the actual number of joints from the robot

        # Define action and observation space
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_joints * 2 + 6 + 4,), dtype=np.float32)
        self.reset()

    def setup_robot(self):
        """Load the robot and initialize joints."""
        robot_id = p.loadURDF("./stridebot.urdf", [0, 0, 0.125], useFixedBase=False)
        self.joint_ids = [j for j in range(p.getNumJoints(robot_id))]
        return robot_id

    def reset(self):
        """Reset the environment to the initial state."""
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.loadURDF("plane.urdf")
        self.robot_id = self.setup_robot()
        return self.get_state()

    def step(self, action):
        """Apply the action and return the state, reward, done, and info."""
        self.apply_action(action)
        p.stepSimulation()
        state = self.get_state()
        reward = self.compute_reward(state)
        done = self.is_done(state)
        info = {}
        return state, reward, done, info

    def get_state(self):
        """Retrieve and return the current state of the robot."""
        joint_states = p.getJointStates(self.robot_id, self.joint_ids)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        _, orientation = p.getBasePositionAndOrientation(self.robot_id)
        _, angular_velocity = p.getBaseVelocity(self.robot_id)
        foot_contacts = [int(len(p.getContactPoints(bodyA=self.robot_id, linkIndexA=link_id)) > 0)
                         for link_id in [3, 7, 11, 15]]
        state = np.concatenate([joint_positions, joint_velocities, list(orientation), list(angular_velocity), foot_contacts])
        return state

    def apply_action(self, action):
        """Apply the provided action to the robot."""
        for i, joint_id in enumerate(self.joint_ids):
            p.setJointMotorControl2(self.robot_id, joint_id, p.POSITION_CONTROL, targetPosition=action[i])

    def compute_reward(self, state):
        """Compute and return the reward for the current state."""
        target_pos = [5, 5, 0.5]  # Define your target position
        current_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        distance = np.linalg.norm(np.array(current_pos[:2]) - np.array(target_pos[:2]))
        return -distance  # Negative reward based on distance

    def is_done(self, state):
        """Determine whether the episode is done."""
        position = p.getBasePositionAndOrientation(self.robot_id)[0]
        return position[2] < 0.2  # Consider done if the robot falls or goes below a certain height

    def render(self, mode='human'):
        """Render the environment."""
        pass

    def close(self):
        """Close the environment and clean up."""
        p.disconnect()

# Use the environment
if __name__ == "__main__":
    env = QuadrupedEnv()
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        if done:
            obs = env.reset()
    env.close()
