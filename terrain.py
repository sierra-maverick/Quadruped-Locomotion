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
