import PyKDL
import numpy as np

rot = PyKDL.Rotation(0,-1,0,1,0,0,0,0,1)
rpy = rot.GetRPY()
print(rpy)
