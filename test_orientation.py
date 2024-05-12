import time
import numpy as np
import pybullet as p
import pybullet_data
from src.kinematic_model import robotKinematics
from src.pybullet_debugger import pybulletDebug
from src.gaitPlanner import trotGait
from terrain import create_hilly_terrain, create_staircase
import csv

# Open a new CSV file
data_file = open('sensor_data.csv', 'w', newline='')
data_writer = csv.writer(data_file)
# Write the header row
data_writer.writerow(['Time', 'Euler_X', 'Euler_Y', 'Euler_Z', 'Angular_Vel_X', 'Angular_Vel_Y', 'Angular_Vel_Z', 'FR_Contact', 'FL_Contact', 'BR_Contact', 'BL_Contact'])



def rendering(render):
    """Enable/disable rendering"""
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, render)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, render)

def robot_init(dt, body_pos, fixed=False):
    physicsClient = p.connect(p.GUI)
    rendering(0)
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(
        fixedTimeStep=dt,
        numSolverIterations=100,
        enableFileCaching=0,
        numSubSteps=1,
        solverResidualThreshold=1e-10,
        erp=1e-1,
        contactERP=0.0,
        frictionERP=0.0,
    )
    # Load terrain
    # p.loadURDF("plane.urdf")
    # create_hilly_terrain(physicsClient)
    stairs = create_staircase(p, step_count=5, step_width=1, step_height=0.1, step_depth=0.5)

    # Load robot
    body_id = p.loadURDF("./stridebot.urdf", body_pos, useFixedBase=fixed)
    joint_ids = []
    maxVel = 3.703
    for j in range(p.getNumJoints(body_id)):
        p.changeDynamics(body_id, j, lateralFriction=1e-5, linearDamping=0, angularDamping=0)
        p.changeDynamics(body_id, j, maxJointVelocity=maxVel)
        joint_ids.append(p.getJointInfo(body_id, j))
    rendering(1)
    return body_id, joint_ids

def robot_stepsim(body_id, body_pos, body_orn, body2feet):
    maxForce = 2
    fr_angles, fl_angles, br_angles, bl_angles, body2feet_ = robotKinematics.solve(body_orn, body_pos, body2feet)
    for i in range(3):
        p.setJointMotorControl2(body_id, i, p.POSITION_CONTROL, targetPosition=fr_angles[i], force=maxForce)
        p.setJointMotorControl2(body_id, 4 + i, p.POSITION_CONTROL, targetPosition=fl_angles[i], force=maxForce)
        p.setJointMotorControl2(body_id, 8 + i, p.POSITION_CONTROL, targetPosition=br_angles[i], force=maxForce)
        p.setJointMotorControl2(body_id, 12 + i, p.POSITION_CONTROL, targetPosition=bl_angles[i], force=maxForce)
    p.stepSimulation()
    return body2feet_

def get_imu_data(body_id):
    _, orientation = p.getBasePositionAndOrientation(body_id)
    _, angular_velocity = p.getBaseVelocity(body_id)
    euler_angles = p.getEulerFromQuaternion(orientation)
    return euler_angles, angular_velocity

def check_foot_contacts(body_id, foot_link_indices):
    contacts = {}
    for foot_link_id in foot_link_indices:
        contact_points = p.getContactPoints(bodyA=body_id, linkIndexA=foot_link_id)
        contacts[foot_link_id] = len(contact_points) > 0
    return contacts

def robot_quit():
    p.disconnect()

if __name__ == '__main__':
    dT = 0.005
    bodyId, jointIds = robot_init(dt=dT, body_pos=[0,0,0.5], fixed=False)
    pybulletDebug = pybulletDebug()
    robotKinematics = robotKinematics()
    trot = trotGait()

    foot_link_ids = [3, 7, 11, 15]  # Foot link indices
    bodytoFeet0 = np.matrix([[0.09, -0.075, 0.1], [0.09, 0.075, 0.1], [-0.09, -0.075, 0.1], [-0.09, 0.075, 0.1]])

for k_ in range(50000):
    current_time = k_ * dT
    pos, orn, L, angle, Lrot, T, sda, offset = pybulletDebug.cam_and_robotstates(bodyId)
    imu_data = get_imu_data(bodyId)
    foot_contacts = check_foot_contacts(bodyId, foot_link_ids)
    bodytoFeet = trot.loop(0.45, angle, Lrot, T, offset, bodytoFeet0, sda)
    robot_stepsim(bodyId, pos, orn, bodytoFeet)

    # Log sensor data
    data_writer.writerow([
        current_time,
        imu_data[0][0], imu_data[0][1], imu_data[0][2],  # Euler angles
        imu_data[1][0], imu_data[1][1], imu_data[1][2],  # Angular velocity
        foot_contacts[3], foot_contacts[7], foot_contacts[11], foot_contacts[15]  # Foot contact states
    ])

data_file.close()


robot_quit()
