# Quadruped-Locomotion
Designing agile and dynamic locomotion for quadrupedal robots presents one of the most formidable challenges in the field of robotics, primarily due to the complexities involved in achieving motion that mimics the adaptability of natural organisms.
This project investigates the enhancement of four-legged robot mobility across challenging terrains, leveraging the principles of Deep Reinforcement Learning (DRL) to bridge the gap between robotic capabilities and biological locomotion. 
The simulation environment for the quadruped robot is created using PyBullet, OpenAI Gym, and Stable Baselines
3 (SB3), providing a robust framework for testing and development. Utilizing the Soft Actor Critic (SAC) Algorithm on-policy Proximal Policy Optimization (PPO) algorithm, the robot autonomously learns from its interactions within this environment, enabling it to adapt and navigate through a variety of complex and unpredictable terrains. 
This simulation-based study revealed that both SAC and PPO are approaching the optimal policy, however they require a significant amount of episodes
to learn (upwards of 5 million).
From this we can infer that with enough computational power, the Quadruped will be able to locomote in a stable and agile manner in any unknown environment.
