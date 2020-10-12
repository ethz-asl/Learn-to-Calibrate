import numpy as np
import PyKDL

T_lr = np.asarray([[1,0,0,0],[0,1,0,-0.06],[0,0,1,0],[0,0,0,1]])
T_lc = np.asarray([[-1,0,0,0],[0,1,0,0],[0,0,-1,0.1],[0,0,0,1]])
print(np.dot(np.linalg.inv(T_lc),T_lr))

rpy = np.asarray([0.06,0.0,-0.1,0,0,1.5708])
rot = PyKDL.Rotation(0.,-1.,0.,1.,0.,0.,0.,0.,1.)
rpy = rot.GetRPY()
A = np.asarray([[0.,-1.,0.,0.06],[1.,0.,0.0,0.0],[0.,0.,1.,-0.1],[0,0,0,1]])
print(A)
print(rpy)

