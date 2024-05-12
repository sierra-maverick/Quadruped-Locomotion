import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Configure logging
logging.basicConfig(filename='quadruped_env.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class QuadrupedEnv(gym.Env):
    """Custom Environment for Quadruped that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(QuadrupedEnv, self).__init__()
        logging.info("Initializing the environment")

        # Connect to PyBullet
        self.physicsClient = p.connect(p.GUI)
        self.robot_id, self.joint_ids = self.setup_robot()  # Ensure joint_ids are retrieved here
        num_joints = len(self.joint_ids)
        logging.info(f"Number of joints: {num_joints}")

        # Define action and observation space
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(43,), dtype=np.float32)
        self.reset()

    def setup_robot(self):
        """Load the robot and initialize joints."""
        try:
            robot_id = p.loadURDF("./stridebot.urdf", [0, 0, 0.125], useFixedBase=False)
            joint_ids = [j for j in range(p.getNumJoints(robot_id))]
            logging.info("Robot loaded successfully with joints initialized")
        except Exception as e:
            logging.error(f"Failed to load robot: {e}")
            raise
        return robot_id, joint_ids

    def reset(self):
        """Reset the environment to the initial state."""
        logging.debug("Resetting the environment")
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.robot_id, self.joint_ids = self.setup_robot()
        return self.get_state()

    def step(self, action):
        """Apply the action and return the state, reward, done, and info."""
        logging.debug("Applying action")
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

        logging.debug(f"Joint positions: {len(joint_positions)}, Joint velocities: {len(joint_velocities)}")
        logging.debug(f"Orientation: {len(orientation)}, Angular velocity: {len(angular_velocity)}")
        logging.debug(f"Foot contacts: {len(foot_contacts)}")

        state = np.concatenate([joint_positions, joint_velocities, list(orientation), list(angular_velocity), foot_contacts])
        logging.debug(f"State retrieved: {state}")
        return state

    def apply_action(self, action):
        """Apply the provided action to the robot."""
        logging.debug(f"Applying actions: {action}")
        for i, joint_id in enumerate(self.joint_ids):
            p.setJointMotorControl2(self.robot_id, joint_id, p.POSITION_CONTROL, targetPosition=action[i])

    def compute_reward(self, state):
        """Compute and return the reward for the current state."""
        target_pos = [5, 5, 0.5]  # Define your target position
        current_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        distance = np.linalg.norm(np.array(current_pos[:2]) - np.array(target_pos[:2]))
        reward = -distance
        logging.debug(f"Reward computed: {reward}")
        return reward

    def is_done(self, state):
        """Determine whether the episode is done."""
        position = p.getBasePositionAndOrientation(self.robot_id)[0]
        done = position[2] < 0.2
        logging.debug(f"Check if done: {done}")
        return done

    def render(self, mode='human'):
        """Render the environment."""
        pass

    def close(self):
        """Close the environment and clean up."""
        logging.info("Closing the environment")
        p.disconnect()

# Main code to use the environment
if __name__ == "__main__":
    env = DummyVecEnv([lambda: QuadrupedEnv()])  # Wrap the environment
    def lr_schedule(step):
        initial_lr = 0.0003
        decay_rate = 0.1
        return initial_lr * (1 - decay_rate * step)

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=lr_schedule)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("ppo_quadruped")

    # Load and test the model
    model = PPO.load("ppo_quadruped", env=env)
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        if done:
            obs = env.reset()
    env.close()
