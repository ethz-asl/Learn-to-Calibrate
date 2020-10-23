#!/usr/bin/env python
# The camera calibration part is based on https://github.com/ros-perception/image_pipeline.

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
import cv2
from scipy import stats
from std_msgs.msg import String
from franka_cal_sim.srv import recordSrv, estimateSrv, recordSrvResponse, estimateSrvResponse

from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge
from random import sample
import subprocess, shlex, signal
import os
import actionlib
import nodelet_rosbag.msg
from rosbag_recorder.srv import RecordTopics, StopRecording
import PyKDL
import subprocess
from subprocess import check_call, CalledProcessError
from pympler import muppy, summary
import gc
import sys
import time

from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion
from std_srvs.srv import Empty
from controller_manager_msgs.srv import ListControllers, UnloadController, LoadController, ReloadControllerLibraries
from enum import Enum

bridge = CvBridge()
#define data structures
#the data acquired in current trajectory
image_data = []
imu_data = []
state_data = []
step = []
num_step = 0
##commands
# #rosbag
# command = "rosbag record -O /home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/kalibr/data.bag simple_camera/image_raw imu_real"
# command = shlex.split(command)
# rm_com = "rm /home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/kalibr/data.bag"
# rm_com = shlex.split(rm_com)
# rein_com = "rosbag reindex /home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/kalibr/data.bag"
# rein_com = shlex.split(rein_com)
# rosbag_proc = subprocess.Popen(command)
# #kalibr command
k_command = "kalibr_calibrate_imu_camera --bag data.bag --cam camchain.yaml --imu imu.yaml --target april_6x6_80x80cm.yaml --max-iter 1 --dont-show-report"
k_command_total = "kalibr_calibrate_imu_camera --bag data.bag --cam camchain.yaml --imu imu.yaml --target april_6x6_80x80cm.yaml --max-iter 3 --dont-show-report"
spawn_command = "rosrun gazebo_ros spawn_model -param robot_description -Y 3.1415926 -urdf -model panda"
launch_command = "roslaunch franka_cal_sim spawn.launch"
launch_0_command = "roslaunch franka_cal_sim spawn_0.launch"
launch_2_command = "roslaunch franka_cal_sim spawn_2.launch"
kill_moveit_command = "rosnode kill controller_spawner"
unspawn_controller_command = "rosrun controller_manager unspawner joint_state_controller panda_hand_controller panda_arm_controller"
unload_controller = "rosrun controller_manager controller_manager unload joint_state_controller panda_hand_controller panda_arm_controller"
stop_controller = "rosrun controller_manager controller_manager stop joint_state_controller panda_hand_controller panda_arm_controller"
start_controller = "rosrun controller_manager controller_manager start joint_state_controller panda_hand_controller panda_arm_controller"
list_controller = "rosrun controller_manager controller_manager list"
#k_command = shlex.split(k_command)

##helper functions
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
ipts = []
opts = []
# db is list of (parameters, image) samples for use in calibration. parameters has form
# (X, Y, size, skew) all normalized to [0,1], to keep track of what sort of samples we've taken
# and ensure enough variety.
db = []
# For each db sample, we also record the detected corners.
good_corners = []
# Set to true when we have sufficiently varied samples to calibrate
last_frame_corners = None
# If the number of samples in db is larger than sample_num, then good enough.
sample_num = 5
# If each param progress in db is larger than progress_value, then good enough.
progress_value = 0.7
# If samples in db are good enough, then it would be True.
goodenough = False
imu_good_enough = False
# Used in compute_goodenough
param_ranges = [0.7, 0.7, 0.4, 0.5]
# variables to measure camera coverage
max_x = 0
max_y = 0
min_x = 0
min_y = 0
obs_n = 0
cov_n = 0
record = 0
if_load = 1
entropy = 0

# # Supported camera models
# class CAMERA_MODEL(Enum):
#     PINHOLE = 0
#     FISHEYE = 1


class ChessboardInfo(object):
    def __init__(self, n_cols=0, n_rows=0, dim=0.0):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.dim = dim


##subscribe topics
def image_callback(data):
    # convert ros msg to cv bridge
    img = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    image_data.append(img)


def imu_callback(data):
    imu_data.append(data)


def state_callback(data):
    state_data.append(data)


def record_callback(req):

    del image_data[:]
    del imu_data[:]
    del state_data[:]
    return True


# Make all private!!!!!
def lmin(seq1, seq2):
    """ Pairwise minimum of two sequences """
    return [min(a, b) for (a, b) in zip(seq1, seq2)]


def lmax(seq1, seq2):
    """ Pairwise maximum of two sequences """
    return [max(a, b) for (a, b) in zip(seq1, seq2)]


def _pdist(p1, p2):
    """
    Distance bwt two points. p1 = (x, y), p2 = (x, y)
    """
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def _get_outside_corners(corners, board):
    """
    Return the four corners of the board as a whole, as (up_left, up_right, down_right, down_left).
    """
    xdim = board.n_cols
    ydim = board.n_rows

    if corners.shape[1] * corners.shape[0] != xdim * ydim:
        raise Exception("Invalid number of corners! %d corners. X: %d, Y: %d" %
                        (corners.shape[1] * corners.shape[0], xdim, ydim))

    up_left = corners[0, 0]
    up_right = corners[xdim - 1, 0]
    down_right = corners[-1, 0]
    down_left = corners[-xdim, 0]

    return (up_left, up_right, down_right, down_left)


def _get_skew(corners, board):
    """
    Get skew for given checkerboard detection.
    Scaled to [0,1], which 0 = no skew, 1 = high skew
    Skew is proportional to the divergence of three outside corners from 90 degrees.
    """
    # TODO Using three nearby interior corners might be more robust, outside corners occasionally
    # get mis-detected
    up_left, up_right, down_right, _ = _get_outside_corners(corners, board)

    def angle(a, b, c):
        """
        Return angle between lines ab, bc
        """
        ab = a - b
        cb = c - b
        return math.acos(
            np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)))

    skew = min(1.0,
               2. * abs((math.pi / 2.) - angle(up_left, up_right, down_right)))
    return skew


def _get_area(corners, board):
    """
    Get 2d image area of the detected checkerboard.
    The projected checkerboard is assumed to be a convex quadrilateral, and the area computed as
    |p X q|/2; see http://mathworld.wolfram.com/Quadrilateral.html.
    """
    (up_left, up_right, down_right,
     down_left) = _get_outside_corners(corners, board)
    a = up_right - up_left
    b = down_right - up_right
    c = down_left - down_right
    p = b + c
    q = a + b
    return abs(p[0] * q[1] - p[1] * q[0]) / 2.


def _get_corners(img, board, refine=True, checkerboard_flags=0):
    """
    Get corners for a particular chessboard for an image
    """
    h = img.shape[0]
    w = img.shape[1]
    if len(img.shape) == 3 and img.shape[2] == 3:
        mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        mono = img
    (ok, corners) = cv2.findChessboardCorners(
        mono, (board.n_cols, board.n_rows),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
        | checkerboard_flags)
    if not ok:
        return (ok, corners)

    # If any corners are within BORDER pixels of the screen edge, reject the detection by setting ok to false
    # NOTE: This may cause problems with very low-resolution cameras, where 8 pixels is a non-negligible fraction
    # of the image size. See http://answers.ros.org/question/3155/how-can-i-calibrate-low-resolution-cameras
    BORDER = 8
    if not all([(BORDER < corners[i, 0, 0] <
                 (w - BORDER)) and (BORDER < corners[i, 0, 1] < (h - BORDER))
                for i in range(corners.shape[0])]):
        ok = False

    # Ensure that all corner-arrays are going from top to bottom.
    if board.n_rows != board.n_cols:
        if corners[0, 0, 1] > corners[-1, 0, 1]:
            corners = np.copy(np.flipud(corners))
    else:
        direction_corners = (corners[-1] - corners[0]) >= np.array([[0.0, 0.0]
                                                                    ])

        if not np.all(direction_corners):
            if not np.any(direction_corners):
                corners = np.copy(np.flipud(corners))
            elif direction_corners[0][0]:
                corners = np.rot90(
                    corners.reshape(board.n_rows, board.n_cols,
                                    2)).reshape(board.n_cols * board.n_rows, 1,
                                                2)
            else:
                corners = np.rot90(
                    corners.reshape(board.n_rows, board.n_cols, 2),
                    3).reshape(board.n_cols * board.n_rows, 1, 2)

    if refine and ok:
        # Use a radius of half the minimum distance between corners. This should be large enough to snap to the
        # correct corner, but not so large as to include a wrong corner in the search window.
        min_distance = float("inf")
        for row in range(board.n_rows):
            for col in range(board.n_cols - 1):
                index = row * board.n_rows + col
                min_distance = min(
                    min_distance,
                    _pdist(corners[index, 0], corners[index + 1, 0]))
        for row in range(board.n_rows - 1):
            for col in range(board.n_cols):
                index = row * board.n_rows + col
                min_distance = min(
                    min_distance,
                    _pdist(corners[index, 0], corners[index + board.n_cols,
                                                      0]))
        radius = int(math.ceil(min_distance * 0.5))
        cv2.cornerSubPix(
            mono, corners, (radius, radius), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    return (ok, corners)


def _get_dist_model(dist_params, cam_model):
    # Select dist model
    if CAMERA_MODEL.PINHOLE == cam_model:
        if dist_params.size > 5:
            dist_model = "rational_polynomial"
        else:
            dist_model = "plumb_bob"
    elif CAMERA_MODEL.FISHEYE == cam_model:
        dist_model = "fisheye"
    else:
        dist_model = "unknown"
    return dist_model


def get_parameters(corners, board, size):
    """
    Return list of parameters [X, Y, size, skew] describing the checkerboard view.
    """
    (width, height) = size
    Xs = corners[:, :, 0]
    Ys = corners[:, :, 1]
    area = _get_area(corners, board)
    border = math.sqrt(area)
    # For X and Y, we "shrink" the image all around by approx. half the board size.
    # Otherwise large boards are penalized because you can't get much X/Y variation.
    p_x = min(1.0, max(0.0, (np.mean(Xs) - border / 2) / (width - border)))
    p_y = min(1.0, max(0.0, (np.mean(Ys) - border / 2) / (height - border)))
    p_size = math.sqrt(area / (width * height))
    skew = _get_skew(corners, board)
    params = [p_x, p_y, p_size, skew]
    return params


def is_slow_moving(corners, last_frame_corners):
    """
    Returns true if the motion of the checkerboard is sufficiently low between
    this and the previous frame.
    """
    max_chessboard_speed = -1.0
    # If we don't have previous frame corners, we can't accept the sample
    if last_frame_corners is None:
        return False
    num_corners = len(corners)
    corner_deltas = (corners - last_frame_corners).reshape(num_corners, 2)
    # Average distance travelled overall for all corners
    average_motion = np.average(np.linalg.norm(corner_deltas, axis=1))
    return average_motion <= max_chessboard_speed


def is_good_sample(db, params, corners, last_frame_corners):
    """
    Returns true if the checkerboard detection described by params should be added to the database.
    """
    max_chessboard_speed = -1
    if not db:
        return True

    def param_distance(p1, p2):
        return sum([abs(a - b) for (a, b) in zip(p1, p2)])

    db_params = [sample[0] for sample in db]
    d = min([param_distance(params, p) for p in db_params])
    #print "d = %.3f" % d #DEBUG
    # TODO What's a good threshold here? Should it be configurable?
    if d <= 0.15:  #0.2
        return False

    if max_chessboard_speed > 0:
        if not is_slow_moving(corners, last_frame_corners):
            return False

    # All tests passed, image should be good for calibration
    return True


def compute_goodenough(db):
    global sample_num
    global progress_value
    if not db:
        return None
    _param_names = ["X", "Y", "Size", "Skew"]
    # Find range of checkerboard poses covered by samples in database
    all_params = [sample[0] for sample in db]
    min_params = all_params[0]
    max_params = all_params[0]
    for params in all_params[1:]:
        min_params = lmin(min_params, params)
        max_params = lmax(max_params, params)
    # Don't reward small size or skew
    min_params = [min_params[0], min_params[1], 0., 0.]

    # For each parameter, judge how much progress has been made toward adequate variation
    # Normalize to [0,1]
    progress = [
        min((hi - lo) / r, 1.0)
        for (lo, hi, r) in zip(min_params, max_params, param_ranges)
    ]
    # If we have lots of samples, allow calibration even if not all parameters are green
    # TODO Awkward that we update self.goodenough instead of returning it
    # goodenough = (len(db) >= 40) or all([p == 1.0 for p in progress])
    goodenough = (len(db) >= sample_num) or all(
        [(p >= progress_value or p == 1.0) for p in progress])
    return list(zip(_param_names, min_params, max_params,
                    progress)), goodenough


def mk_object_points(boards, use_board_size=False):
    opts = []
    for i, b in enumerate(boards):
        num_pts = b.n_cols * b.n_rows
        opts_loc = np.zeros((num_pts, 1, 3), np.float32)
        for j in range(num_pts):
            opts_loc[j, 0, 0] = (j / b.n_cols)
            opts_loc[j, 0, 1] = (j % b.n_cols)
            opts_loc[j, 0, 2] = 0
            if use_board_size:
                opts_loc[j, 0, :] = opts_loc[j, 0, :] * b.dim
        opts.append(opts_loc)
    return opts


@staticmethod
def linear_error(corners, b):
    """
    Returns the linear error for a set of corners detected in the unrectified image.
    """

    if corners is None:
        return None

    def pt2line(x0, y0, x1, y1, x2, y2):
        """ point is (x0, y0), line is (x1, y1, x2, y2) """
        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) *
                   (y2 - y1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    cc = b.n_cols
    cr = b.n_rows
    errors = []
    for r in range(cr):
        (x1, y1) = corners[(cc * r) + 0, 0]
        (x2, y2) = corners[(cc * r) + cc - 1, 0]
        for i in range(1, cc - 1):
            (x0, y0) = corners[(cc * r) + i, 0]
            errors.append(pt2line(x0, y0, x1, y1, x2, y2))
    if errors:
        return math.sqrt(sum([e**2 for e in errors]) / len(errors))
    else:
        return None


def do_calibration(req, db, good_corners, cam_model):
    if not good_corners:
        print("**** Collecting corners for all images! ****")  #DEBUG
        images = [i for (p, i) in db]
        good_corners = good_corners
    size = (db[0][1].shape[1], db[0][1].shape[0]
            )  # TODO Needs to be set externally

    # indicate camera model
    if cam_model == 0:
        camera_model = CAMERA_MODEL.PINHOLE
    elif cam_model == 1:
        camera_model = CAMERA_MODEL.FISHEYE
    else:
        raise Exception('Please indicate camera model!')

    ret, mtx, dist, rvecs, tvecs, ipts, opts = cal_fromcorners(
        req, good_corners, camera_model, size)

    # if camera_model == CAMERA_MODEL.FISHEYE:
    #     print("mono fisheye calibration...")
    #     # WARNING: cv2.fisheye.calibrate wants float64 points
    #     ipts64 = numpy.asarray(ipts, dtype=numpy.float64)
    #     ipts = ipts64
    #     opts64 = numpy.asarray(opts, dtype=numpy.float64)
    #     opts = opts64
    #     reproj_err, self.intrinsics, self.distortion, rvecs, tvecs = cv2.fisheye.calibrate(
    #         opts, ipts, self.size,
    #         intrinsics_in, None, flags = self.fisheye_calib_flags)
    return ret, mtx, dist, rvecs, tvecs, ipts, opts


def cal_fromcorners(req, good, camera_model, size):
    """
    :param good: Good corner positions and boards
    :type good: [(corners, ChessboardInfo)]
    """
    # termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10,
                    0.001)
    boards = [b for (_, b) in good]
    ipts = [points for (points, _) in good]
    opts = mk_object_points(boards)
    # If FIX_ASPECT_RATIO flag set, enforce focal lengths have 1/1 ratio
    # camera intrinsic matrix
    mtx = np.eye(3, dtype=np.float64)
    if len(req.params) > 3:
        mtx[0, 0] = req.params[0]
        mtx[1, 1] = req.params[1]
        mtx[0, 2] = req.params[2]
        mtx[1, 2] = req.params[3]
    mtx[2, 2] = 1

    if camera_model == CAMERA_MODEL.PINHOLE:
        ret, mtx, dist_not_flatten, rvecs, tvecs = cv2.calibrateCamera(
            opts,
            ipts,
            size,
            cameraMatrix=mtx,
            distCoeffs=None,
            rvecs=None,
            tvecs=None,
            flags=0,
            criteria=criteria_cal)
        dist = dist_not_flatten.flat[:8].reshape(-1, 1)
    elif camera_model == CAMERA_MODEL.FISHEYE:
        print("mono fisheye calibration...")
        # WARNING: cv2.fisheye.calibrate wants float64 points
        ipts64 = np.asarray(ipts, dtype=np.float64)
        ipts = ipts64
        opts64 = np.asarray(opts, dtype=np.float64)
        opts = opts64
        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            opts, ipts, size, mtx, None, flags=0, criteria=criteria_cal)
    else:
        raise Exception('Please indicate camera model!')
    return ret, mtx, dist, rvecs, tvecs, ipts, opts


def camera_intrinsic_calibration(req, image_data):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10,
                    0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    for img in image_data:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            # Once we find the corners, we can increase their accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)
            imgpoints.append(corners2)

    # index list for imgpoints
    list1 = np.arange(0, len(imgpoints), 1)
    # camera intrinsic matrix
    mtx = np.zeros((3, 3))
    if len(req.params) > 3:
        mtx[0, 0] = req.params[0]
        mtx[1, 1] = req.params[1]
        mtx[0, 2] = req.params[2]
        mtx[1, 2] = req.params[3]
    mtx[2, 2] = 1

    # optimize data step by step based on sampled imgs, get best one
    min_error = 1000
    best_mtx = mtx
    for i in range(10):
        cur_data = list1
        if len(imgpoints) > 20:
            # randomly select 20 keypoints to do calibration
            cur_data = sample(list1, 20)
        cur_obj = list(objpoints[i] for i in cur_data)
        cur_img = list(imgpoints[i] for i in cur_data)
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                cur_obj,
                cur_img,
                gray.shape[::-1],
                cameraMatrix=mtx,
                distCoeffs=None,
                rvecs=None,
                tvecs=None,
                flags=0,
                criteria=criteria_cal)
        except:
            gray = cv2.cvtColor(image_data[0], cv2.COLOR_BGR2GRAY)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                cur_obj,
                cur_img,
                gray.shape[::-1],
                cameraMatrix=mtx,
                distCoeffs=None,
                rvecs=None,
                tvecs=None,
                flags=0,
                criteria=criteria_cal)

        #evaluate
        tot_error = 0
        mean_error = 0
        for j in range(len(cur_obj)):
            imgpoints2, _ = cv2.projectPoints(cur_obj[j], rvecs[j], tvecs[j],
                                              mtx, dist)
            error = cv2.norm(cur_img[j], imgpoints2,
                             cv2.NORM_L2) / len(imgpoints2)
            tot_error += error
        mean_error = tot_error / len(cur_obj)
        if mean_error < min_error:
            min_error = mean_error
            best_mtx = mtx

        rospy.loginfo(rospy.get_caller_id() + 'I get corners %s',
                      len(imgpoints))
        rospy.loginfo(rospy.get_caller_id() + 'I get parameters %s',
                      best_mtx[0, 0])
    return imgpoints, best_mtx


def camera_intrinsic_calibration2(req, image_data):
    # db is list of (parameters, image) samples for use in calibration. parameters has form
    # (X, Y, size, skew) all normalized to [0,1], to keep track of what sort of samples we've taken
    # and ensure enough variety.
    global db
    # For each db sample, we also record the detected corners.
    global good_corners
    # Set to true when we have sufficiently varied samples to calibrate
    global last_frame_corners
    global goodenough
    global param_ranges
    global ipts
    global opts

    # image directory
    images = image_data
    boards = []
    options_square = []
    options_size = []
    # info about chessboard
    for i in range(3):
        options_square.append("0.01")
        # width x height
        options_size.append("6x7")
    # print(options_size)
    # print(options_square)
    for (sz, sq) in zip(options_size, options_square):
        size = tuple([int(c) for c in sz.split('x')])
        boards.append(ChessboardInfo(size[0], size[1], float(sq)))
    # board = [
    #     ChessboardInfo(i.n_cols, i.n_rows, i.dim)
    #     for i in boards
    # # ]
    board = [
        ChessboardInfo(max(i.n_cols, i.n_rows), min(i.n_cols, i.n_rows), i.dim)
        for i in boards
    ]
    rospy.loginfo(rospy.get_caller_id() + ' ' + 'The column of board %s',
                  board[1].n_cols)
    rospy.loginfo(rospy.get_caller_id() + ' ' + 'The row of board %s',
                  board[1].n_rows)
    # print(board[1].n_cols)

    for img in images:
        # Find the chess board corners
        (ok, corners) = _get_corners(img,
                                     board[1],
                                     refine=True,
                                     checkerboard_flags=0)
        '''
        # just to verify the method is correct
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners2 = cv2.findChessboardCorners(gray, (17,12),None)
        corners3 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        if np.all([corners,corners3]):
            print("True")
        else:
            print("False")
        '''
        progress = [0, 0, 0, 0]
        # If found, add object points, image points (after refining them)
        #rospy.loginfo(rospy.get_caller_id() + 'ok: %s', ok)
        if ok:
            # skew = _get_skew(corners2, board[1])
            # print(skew)
            # need to check again!
            h = img.shape[0]
            w = img.shape[1]
            size = [w, h]
            # Return list of parameters [X, Y, size, skew] describing the checkerboard view.
            params = get_parameters(corners, board[1], size)
            # print(params)
            # Add sample to database only if it's sufficiently different from any previous sample.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if is_good_sample(db, params, corners, last_frame_corners):
                db.append((params, gray))
                #print(params)
                good_corners.append((corners, board[1]))
                # print((
                #         "*** Added sample %d, p_x = %.3f, p_y = %.3f, p_size = %.3f, skew = %.3f"
                #         % tuple([len(db)] + params)))
            # update the frame corners
            last_frame_corners = corners
            # Find range of checkerboard poses covered by samples in database
            # progress: For each parameter, judge how much progress has been made toward adequate variation
            # If we have lots of samples (>5), allow calibration even if not all parameters are good enough (p >= 0.7 for p in progress).
            # return list(zip(_param_names, min_params, max_params, progress)), goodenough
            zip_params, goodenough = compute_goodenough(db)
            #get progress
            progress = [par[3] for par in zip_params]
            # print(zip_params)
            # print(goodenough)
        else:
            pass

    mtx = np.eye(3, dtype=np.float64)
    if len(req.params) > 3:
        mtx[0, 0] = req.params[0]
        mtx[1, 1] = req.params[1]
        mtx[0, 2] = req.params[2]
        mtx[1, 2] = req.params[3]
    mtx[2, 2] = 1

    reproj_err = 1
    if goodenough:
        # If good enough, use db and good_corners to do calibration.
        # The output are:
        # ret: reproj_err, mtx: intrinsics,
        # dist: dist_coeffs = None, rvecs=None, tvecs=None,
        # ipts: imgpoints, 2d points in image plane,
        # opts: objpoints, 3d point in real world space.
        ret, mtx, dist, rvecs, tvecs, ipts, opts = do_calibration(req,
                                                                  db,
                                                                  good_corners,
                                                                  cam_model=0)
        reproj_err = ret
        print(mtx)
        # params update
        # req.par_upd = [mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]]
        # print(req.par_upd)
    '''
    # just for debug 
    imgpoints, objpoints, best_mtx = camera_intrinsic_calibration_ver1(images)
    print(best_mtx)
    '''
    return mtx, ipts, opts, progress, reproj_err


def compute_coverage(res, imgpoints):
    global max_x
    global max_y
    global min_x
    global min_y
    img_flat = np.reshape(np.asarray(imgpoints), (-1, 3))
    res.coverage = np.max(img_flat, axis=0)[0] - max_x - np.min(
        img_flat, axis=0)[0] + min_x
    res.coverage = res.coverage + np.max(img_flat, axis=0)[1] - max_y - np.min(
        img_flat, axis=0)[0] + min_y
    max_x = np.max(img_flat, axis=0)[0]
    max_y = np.max(img_flat, axis=0)[1]
    min_x = np.min(img_flat, axis=0)[0]
    min_y = np.min(img_flat, axis=0)[1]
    return res


def compute_obsevation(res, imu_data):
    res.obs = 0
    return res


def spawn_model(spawn_flag):
    global if_load
    kalibr_path = rospy.get_param("/rl_client/kalibr_path")
    sensor_path = rospy.get_param("/rl_client/sensor_gazebo_path")
    hand_path = rospy.get_param("/rl_client/hand_path")
    cam_chain_path = kalibr_path + "camchain.yaml"
    model_name = "panda"
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    list_controller_srv = rospy.ServiceProxy(
        '/controller_manager/list_controllers', ListControllers)
    load_controller_srv = rospy.ServiceProxy(
        '/controller_manager/load_controller', LoadController)
    unload_controller_srv = rospy.ServiceProxy(
        '/controller_manager/unload_controller', UnloadController)
    reload_controller_srv = rospy.ServiceProxy(
        '/controller_manager/reload_controller_libraries',
        ReloadControllerLibraries)
    spawn_urdf_command = "rosrun gazebo_ros spawn_model -file " + sensor_path + " -Y 3.1415926 -urdf -model panda"
    if spawn_flag:
        # generate new model
        #generate new hov:
        centor_hov = 1.0
        centor_ext = np.asarray([0.0, 0.0, 0.0584, 0, 0, 0])
        sample_hov = np.random.normal(centor_hov, 0.05)
        sample_ext = np.random.normal(
            centor_ext, np.asarray([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]))
        #first modify urdf

        #modify intrinsic
        my_file = open(sensor_path)
        #read file
        string_list = my_file.readlines()
        my_file.close()
        new_string_list = []

        for line in string_list:
            if line[0:16] == "<horizontal_fov>":
                line = "<horizontal_fov>" + str(
                    sample_hov) + "</horizontal_fov>\n"
                print(line)
            new_string_list.append(line)

        #write the file again
        my_file = open(sensor_path, "w")
        new_file_contents = "".join(new_string_list)
        my_file.write(new_file_contents)
        my_file.close()

        #modify extrinsic
        my_file = open(hand_path)
        #read file
        string_list = my_file.readlines()
        my_file.close()
        new_string_list = []

        for i in range(len(string_list)):
            line = string_list[i]
            if i == 73:
                line = "<origin xyz=\"" + str(sample_ext[0]) + " "
                line += str(sample_ext[1]) + " "
                line += str(sample_ext[2]) + "\" rpy=\""
                line += str(sample_ext[3]) + " "
                line += str(sample_ext[4]) + " "
                line += str(sample_ext[5]) + "\"/>\n"
                print(line)
            new_string_list.append(line)

        #write the file again
        my_file = open(hand_path, "w")
        new_file_contents = "".join(new_string_list)
        my_file.write(new_file_contents)
        my_file.close()

        #modify yaml file
        #compute intrinsics
        fx = float(320.0 / np.tan(sample_hov / 2))
        intrinsic = [fx, fx, 320.5, 240.5]
        my_file = open(cam_chain_path)
        #read file
        string_list = my_file.readlines()
        my_file.close()
        new_string_list = []
        #modify the sensor line
        for line in string_list:
            if line[0:12] == "  intrinsics":
                line = "  intrinsics: " + str(intrinsic) + "\n"
                print(line)
            new_string_list.append(line)

        #write the file again
        my_file = open(cam_chain_path, "w")
        new_file_contents = "".join(new_string_list)
        my_file.write(new_file_contents)
        my_file.close()

        #broacast new params
        rospy.set_param("/rl_client/imu_cam_ground_truth",
                        sample_ext.astype(float).tolist())
        rospy.set_param("/rl_client/cam_ground_truth", intrinsic)

        if (if_load % 40 == 1):  #40
            process = subprocess.Popen(launch_command.split(),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

        else:
            process = subprocess.Popen(start_controller.split(),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            process = subprocess.Popen(launch_0_command.split(),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        if_load += 1

    else:

        #call delete model service
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            delete_model_service = rospy.ServiceProxy('/gazebo/delete_model',
                                                      DeleteModel)
            resp1 = delete_model_service(model_name)
            print(resp1.status_message)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

        process = subprocess.Popen(kill_moveit_command.split(),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        # process = subprocess.Popen(kill_moveit_command.split(),
        #                            stdout=subprocess.PIPE,
        #                            stderr=subprocess.PIPE)

        os.system(kill_moveit_command)


def estimate_callback(req):
    global image_data
    global state_data
    global imu_data
    global objpoints
    global imgpoints
    global db
    global obs_n
    global cov_n
    global good_corners
    global last_frame_corners
    global goodenough
    global num_step
    global entropy
    rospy.loginfo(rospy.get_caller_id() + 'I heard image %s', len(image_data))
    rospy.loginfo(rospy.get_caller_id() + 'I heard imu %s', len(imu_data))
    rospy.loginfo(rospy.get_caller_id() + 'I heard state %s', len(state_data))
    local_img_data = image_data
    t = 0
    # indicate camera model
    #     PINHOLE = 0
    #     FISHEYE = 1
    #cam_model = 0

    if req.reset == 1:
        #clear all the record data

        del image_data[:]
        del imu_data[:]
        del state_data[:]
        del objpoints[:]
        del imgpoints[:]
        del good_corners[:]
        del db[:]
        obs_n = 0
        num_step = 0
        last_frame_corners = None
        goodenough = False

        #feed back the update
        res = estimateSrvResponse()
        res.par_upd = [0, 0, 0, 0, 0, 0, 0, 0]
        res.obs = 0
        res.coverage = 0
        return res

    if not req.reset == 1:
        num_step += 1
        if len(image_data) == 0:
            res = estimateSrvResponse()
            res.par_upd = [0, 0, 0, 0, 0, 0, 0, 0]
            res.obs = 0
            res.coverage = 0

            return res
        # estimation
        # camera intrinsic calibration

        # imgpoints, best_mtx = camera_intrinsic_calibration(req, image_data)
        reproj_err = 1
        try:
            best_mtx, ipts, opts, progress, reproj_err = camera_intrinsic_calibration2(
                req, image_data)
            rospy.loginfo(rospy.get_caller_id() + 'I get parameters %s',
                          best_mtx[0, 0])
        except:
            pass

        if num_step >= 4:
            spawn_model(0)  # remove old
            time.sleep(2)
            spawn_model(1)  # generate new
            time.sleep(5)

        # compute the coverage
        rospy.loginfo(rospy.get_caller_id() + 'I get good corners %s',
                      len(good_corners))

        if len(good_corners) > 0:
            res = estimateSrvResponse()
            for c, b in good_corners:
                imgpoints.append(c)  # just for coverage calculation.
            rospy.loginfo(rospy.get_caller_id() + 'I get corners %s',
                          len(imgpoints))

            ####progress measures camera coverage
            res.coverage = np.sum(progress) - cov_n
            cov_n = np.sum(progress)
            # get parameter update
            # compute the observation
            res = compute_obsevation(res, imu_data)
            res.par_upd = [0, 0, 0, 0]

            res.obs = 1.0 * len(db) / 2.0 - obs_n

            if num_step == 4:
                res.par_upd = [
                    best_mtx[0, 0], best_mtx[1, 1], best_mtx[0, 2], best_mtx[1,
                                                                             2]
                ]
                #add reproj
                #res.obs+=5*(1-reproj_err)

            res.par_upd += list(progress)

            obs_n = 1.0 * len(db) / 2.0
            rospy.loginfo(rospy.get_caller_id() + 'I get db %s', len(db))

            rospy.loginfo(rospy.get_caller_id() + 'I get par_upd shape %s',
                          len(res.par_upd))
            rospy.loginfo(rospy.get_caller_id() + 'I get total cov %s',
                          np.sum(progress))
            rospy.loginfo(rospy.get_caller_id() + 'I get reproj err %s',
                          reproj_err)
            return res
        else:
            res = estimateSrvResponse()
            res.obs = 0
            res.coverage = 0
            res.par_upd = [0, 0, 0, 0, 0, 0, 0, 0]
            return res


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('/simple_camera/image_raw', Image, image_callback)
    rospy.Subscriber('/imu_real', Imu, imu_callback)
    rospy.Subscriber('/gazebo/link_states', LinkStates, state_callback)

    record_s = rospy.Service('record_service', recordSrv, record_callback)
    estimate_s = rospy.Service('estimate_service', estimateSrv,
                               estimate_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
