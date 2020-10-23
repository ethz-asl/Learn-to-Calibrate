#!/usr/bin/env python

from __future__ import print_function
from calibrator import Calibrator, MonoCalibrator, StereoCalibrator, Patterns
from calibrator import CAMERA_MODEL
import math
import time
import sys
import glob
import rosbag
import rospy
import numpy as np
import tensorflow as tf
import cv2
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge
from random import sample
import subprocess, shlex, signal
import os
import actionlib

bridge = CvBridge()

## define data structures
# the data acquired in current trajectory
image_data = []
imu_data = []
state_data = []
step = []
num_step = 0

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
ipts = []
opts = []


# read yaml file or give directly
def load_camera_intrinsics():
    mtx = np.eye(3, dtype=np.float64)
    mtx[0, 0] = 1.8532040955929899e+03
    mtx[1, 1] = 1.8539053769057191e+03
    mtx[0, 2] = 5.3488742500529372e+02
    mtx[1, 2] = 9.3665016534455742e+02
    mtx[2, 2] = 1
    dist = np.array([
        5.3631031595833795e-02, 5.8546496555231975e-02,
        -4.6216798555226001e-04, 1.6745292279350978e-03, 0.
    ])
    return mtx, dist


# it takes the corners in the chessboard (obtained using cv2.findChessboardCorners()) and axis points to draw a 3D axis
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


# initialization func
def camera_pose_init(img, objp, mtx, dist):
    '''
    img = cv2.imread(fname)
    objp = 3d point in real world space
    mtx = K matrix
    dist = distortion
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (17, 12), None)
    if ret == True:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
                    0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria)
        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(
            objp.reshape(-1, 1, 3), corners2.reshape(-1, 1, 2), mtx, dist)
        return rvecs, tvecs
    return None  # or return random initialization


## landmark points in 3d in real world space
# the tranlation vector of landmarks (corners on checkerboard, (w*h, 3)) wrt the world origin, in world coordinate
objp = np.zeros((12 * 17, 3), np.float32)
objp[:, :2] = np.mgrid[0:17, 0:12].T.reshape(-1, 2)

# the axis points are (3,0,0),(0,3,0),(0,0,3) in 3D space.
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
# load camera intrinsics
mtx, dist = load_camera_intrinsics()

### Extrinsic

## IMU pose in world coordinate, from world to IMU
# Quaternion(w, x, y, z)
G_q_GI = np.array([1, 2, 3, 4], np.float32)
# change to rotation matrix
G_q_GI_quat = Quaternion(G_q_GI)
G_R_GI = G_q_GI_quat.rotation_matrix
# translation vector
G_p_GI = np.array([1, 2, 3], np.float32)
# transformation matrix
G_T_GI = np.vstack((np.hstack((G_R_GI, G_p_GI[:, None])), [0, 0, 0, 1]))
# print(G_T_GI)

## camera-IMU extrinsic, from camera to IMU
# Quaternion(w, x, y, z)
q_CI = np.array([1, 2, 3, 4], np.float32)
# change to rotation matrix
q_CI_quat = Quaternion(q_CI)
R_CI = q_CI_quat.rotation_matrix
# translation vector (3x1)
p_CI = np.array([1, 2, 3], np.float32)
# transformation matrix
T_CI = np.vstack((np.hstack((R_CI, p_CI[:, None])), [0, 0, 0, 1]))
# print(T_CI)

## camera pose in world coordinate, from world to camera
G_T_GC = np.dot(G_T_GI, T_CI.T)
G_R_GC = G_T_GC[:3, :3]  #(3x3)
G_p_GC = G_T_GC[:3, 3]  #(3x1)
# print(G_p_GC)
# print(p_CI)

# Find the rotation and translation vectors.
rvec_GC, _ = cv2.Rodrigues(G_R_GC)
tvec_GC = G_p_GC.reshape(3, 1)
# print(tvec)
# print(rvec)

# project landmark points from 3d space to image plane
imgpoints2, jac = cv2.projectPoints(objp, rvec_GC, tvec_GC, mtx, dist)

## Compute reprojection error
# first find the landmark points (corners) on the measured image data
imgpoints = []  # measured 2d points in image plane
objpoints = []  # 3d point in real world space
images = glob.glob('/home/felix/panda_ws/src/franka_cal_sim/python/img4.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (17, 12), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
                    0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria)
        imgpoints.append(corners2)
        print(
            "found the landmark points (corners) on the measured image data!")
imgpoints = np.array(imgpoints)
imgpoints2 = imgpoints2[:, 0, :]
imgpoints = imgpoints[0, :, 0, :]
# print(imgpoints2.shape)
# print(imgpoints.shape)

# Compute reprojection error
error = cv2.norm(imgpoints, imgpoints2, cv2.NORM_L2) / len(objpoints)
print('reprojection error: {}'.format(error))

### Bspline
