import pybullet as p
import numpy as np

def create_hilly_terrain(physicsClient, terrainWidth=256, terrainLength=256, terrainScale=1.0):  # Increased terrainScale
    """Creates a hilly terrain using a generated height map."""
    # Generate height data for a hilly terrain using a sine wave pattern
    heights = np.zeros((terrainWidth, terrainLength), dtype=np.float32)
    for i in range(terrainWidth):
        for j in range(terrainLength):
            heights[i, j] = (np.sin(i / 20.0) * np.cos(j / 20.0)) * terrainScale  # Might adjust the frequency

    # Convert the numpy array to a list
    heightfieldData = heights.flatten().tolist()

    # Create the terrain shape with increased z scaling
    terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, 
                                          meshScale=[0.1, 0.1, 1],  # Increased Z scaling
                                          heightfieldTextureScaling=terrainWidth//2,
                                          heightfieldData=heightfieldData,
                                          numHeightfieldRows=terrainWidth,
                                          numHeightfieldColumns=terrainLength)

    # Create the terrain body
    terrainBody = p.createMultiBody(0, terrainShape)
    p.resetBasePositionAndOrientation(terrainBody, [0, 0, 0], [0, 0, 0, 1])
    return terrainBody


def create_staircase(physicsClient, step_count=5, step_width=1, step_height=0.2, step_depth=0.5):
    """Creates a simple staircase."""
    base_position = [0, 0, 0]  # Starting position of the first step
    stairs = []
    for i in range(step_count):
        step_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[step_width / 2, step_depth / 2, step_height / 2]
        )
        step_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[step_width / 2, step_depth / 2, step_height / 2],
            rgbaColor=[1, 0.6, 0, 1]  # Orange color
        )
        step_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=step_shape,
            baseVisualShapeIndex=step_visual,
            basePosition=[base_position[0], base_position[1], base_position[2] + step_height / 2]
        )
        stairs.append(step_body)
        # Update base_position for the next step
        base_position[2] += step_height  # Increase height
        base_position[1] += step_depth  # Move deeper

    return stairs

