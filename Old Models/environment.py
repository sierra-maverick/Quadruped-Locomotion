import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import SAC

# Initialize PyBullet environment
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

    def reset(self):
        # Reset the environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Load terrain
        self.load_terrain()

        # Load quadruped
        self.robot = p.loadURDF("quadruped/quadruped.urdf", basePosition=[0, 0, 0.5], baseOrientation=p.getQuaternionFromEuler([0, np.pi, 0]))

        # Initialize observation
        observation = self.get_observation()
        return observation

    def load_terrain(self):
        ''' Plane terrain 
        self.terrain = p.loadURDF("plane.urdf")
        ''' 

        '''
        #Hilly terrain
        terrainWidth = 256
        terrainLength = 256
        terrainHeight = 0.5
        heightfieldData = [0]*terrainWidth*terrainLength
        for i in range(terrainLength):
            for j in range(terrainWidth):
                height = np.sin(i / 20.0) * np.cos(j / 20.0) * terrainHeight
                heightfieldData[i*terrainWidth+j] = height
        terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                            meshScale=[0.05, 0.05, 1],
                                            heightfieldTextureScaling=(terrainWidth-1)/2,
                                            heightfieldData=heightfieldData,
                                            numHeightfieldRows=terrainWidth,
                                            numHeightfieldColumns=terrainLength)
        self.terrain = p.createMultiBody(0, terrainShape)

        '''
        
        '''
        #Stairs terrain
        step_height = 0.7
        step_width = 0.5
        step_length = 0.5
        num_steps = 10
        for i in range(num_steps):
            step_shape = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                halfExtents=[step_width / 2, step_length / 2, step_height / 2])
            step_body = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=step_shape,
                                        basePosition=[i * step_width, 0, i * step_height / 2],
                                        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))


        ''' 
        #Random uneven terrain
        terrainWidth = 256
        terrainLength = 256
        terrainHeight = 1.2
        heightPerturbationRange = 0.05
        heightfieldData = [terrainHeight * np.random.uniform(-heightPerturbationRange, heightPerturbationRange) for _ in range(terrainWidth*terrainLength)]
        terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                            meshScale=[0.05, 0.05, 1],
                                            heightfieldTextureScaling=(terrainWidth-1)/2,
                                            heightfieldData=heightfieldData,
                                            numHeightfieldRows=terrainWidth,
                                            numHeightfieldColumns=terrainLength)
        self.terrain = p.createMultiBody(0, terrainShape)

        

    def get_observation(self):
        # Implement observation extraction
        return np.zeros(self.observation_space.shape)

    def step(self, action):
        # Apply action, step simulation, and get observation
        p.stepSimulation()
        observation = self.get_observation()

        # Compute reward and done condition
        reward = self.compute_reward()
        done = self.check_done()

        return observation, reward, done, {}

    def compute_reward(self):
        # Define reward calculation logic
        return 0

    def check_done(self):
        # Define logic to check if episode is done
        return False

    def render(self, mode='human'):
        # Rendering logic
        pass

    def close(self):
        p.disconnect()



# Instantiate the environment
env = QuadrupedEnv()

# Define SAC model
model = SAC("MlpPolicy", env, verbose=1)

# Training loop
for episode in range(1000):  # Number of episodes
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
    print(f"Episode {episode} finished")

import numpy as np

# Save the model
model.save("sac_quadruped")

# Uncomment the lines below to execute the simulation and training process
# env = QuadrupedEnv()
# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=50000)
# model.save("sac_quadruped")