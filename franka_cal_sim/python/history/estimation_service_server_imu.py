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
import pandas as pd
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
from subprocess import check_call,CalledProcessError
from pympler import muppy, summary
import gc
import sys
import time


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
kalibr_path = "/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/kalibr"
k_command = "kalibr_calibrate_imu_camera --bag data.bag --cam camchain.yaml --imu imu.yaml --target checkerboard_7x6_6x6cm.yaml --max-iter 1 --dont-show-report --timeoffset-padding 0.1"
k_command_total = "kalibr_calibrate_imu_camera --bag data.bag --cam camchain.yaml --imu imu.yaml --target checkerboard_7x6_6x6cm.yaml --max-iter 10 --dont-show-report --timeoffset-padding 0.1"
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
sample_num = 1
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
img_cov = 0
total_entropy = 0


class ChessboardInfo(object):
    def __init__(self, n_cols=0, n_rows=0, dim=0.0):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.dim = dim


##subscribe topics
def image_callback(data):
    global record
    if record==1:
        img = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        image_data.append(img)


def imu_callback(data):
    imu_data.append(data)


def state_callback(data):
    state_data.append(data)


def record_callback(req):
    global record
    del image_data[:]
    del imu_data[:]
    del state_data[:]
    record = 1
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
    if d <= 0.2:
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


def do_calibration(req, db, good_corners):
    if not good_corners:
        print("**** Collecting corners for all images! ****")  #DEBUG
        images = [i for (p, i) in db]
        good_corners = good_corners
    size = (db[0][1].shape[1], db[0][1].shape[0]
            )  # TODO Needs to be set externally
    camera_model = CAMERA_MODEL.PINHOLE
    ret, mtx, dist, rvecs, tvecs, ipts, opts = cal_fromcorners(
        req, good_corners, camera_model, size)
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
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(opts,
                                                       ipts,
                                                       size,
                                                       cameraMatrix=mtx,
                                                       distCoeffs=None,
                                                       rvecs=None,
                                                       tvecs=None,
                                                       flags=0,
                                                       criteria=criteria_cal)
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
    progress = [0, 0, 0, 0]
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

        # If found, add object points, image points (after refining them)
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

    if goodenough:
        # If good enough, use db and good_corners to do calibration.
        # The output are:
        # ret: reproj_err, mtx: intrinsics,
        # dist: dist_coeffs = None, rvecs=None, tvecs=None,
        # ipts: imgpoints, 2d points in image plane,
        # opts: objpoints, 3d point in real world space.
        ret, mtx, dist, rvecs, tvecs, ipts, opts = do_calibration(
            req, db, good_corners)
        print(mtx)
        # params update
        # req.par_upd = [mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]]
        # print(req.par_upd)
    '''
    # just for debug 
    imgpoints, objpoints, best_mtx = camera_intrinsic_calibration_ver1(images)
    print(best_mtx)
    '''
    return mtx, ipts, opts, progress


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


def get_kalibr_results():
    #initialize
    err = 1
    ext_lines = np.identity(4)
    for root, dirs, files in os.walk(kalibr_path):
        if 'results-imucam-data.txt' in files:

            mylines = []
            with open(kalibr_path + 'results-imucam-data.txt',
                      'rt') as myfile:  # Open lorem.txt for reading text
                for line in myfile:
                    #print(line)
                    mylines.append(line)

            for i in range(len(mylines)):
                #get reprojection error
                if mylines[i][0:12] == "Reprojection":
                    err_str = mylines[i].split(',')[0].split(' ')
                    err = float(err_str[len(err_str) - 1])

                if mylines[i][0:4] == "T_ci":
                    ext_lines = mylines[i + 1:i + 5]

            #get extrinsics
            for i in range(len(ext_lines)):
                ext_lines[i] = ext_lines[i].rstrip().strip("[]").strip(" [ ").split()
                for j in range(len(ext_lines[i])):
                    ext_lines[i][j] = float(ext_lines[i][j])

            #remove result
            os.chdir(kalibr_path[:-1])
            os.system("rm results-imucam-data.txt")


    return err, ext_lines


def Entropy(labels, base=2.73):
    # Defines the (discrete) distribution
    probs = pd.Series(labels).value_counts() / len(labels)
    # Calculate the entropy of a distribution for given probability values.
    en = stats.entropy(probs, base=base)
    return en


def compute_imu_entropy(req, imu_data):
    '''
    ---
    header:
    seq: 10750
    stamp:
        secs: 53
        nsecs: 755000000
    frame_id: "panda_rightfinger"
    orientation:
    x: -0.707028604474
    y: -0.00559323353604
    z: 0.707139771256
    w: 0.00571070057722
    orientation_covariance: [1.6659725114535607e-07, 0.0, 0.0, 0.0, 1.6659725114535607e-07, 0.0, 0.0, 0.0, 1.1519236000000001e-07]
    angular_velocity:
    x: 9.703539399e-05
    y: 0.000340150013805
    z: 0.000349645491
    angular_velocity_covariance: [1.1519236000000001e-07, 0.0, 0.0, 0.0, 1.1519236000000001e-07, 0.0, 0.0, 0.0, 1.1519236000000001e-07]
    linear_acceleration:
    x: -10.8580385388
    y: -0.0741810464577
    z: -0.275371911387
    linear_acceleration_covariance: [1.6e-05, 0.0, 0.0, 0.0, 1.6e-05, 0.0, 0.0, 0.0, 1.6e-05]

    '''
    orientation_x = []
    orientation_y = []
    orientation_z = []
    orientation_w = []
    angular_velocity_x = []
    angular_velocity_y = []
    angular_velocity_z = []
    linear_acceleration_x = []
    linear_acceleration_y = []
    linear_acceleration_z = []

    for imu in imu_data:
        orientation_x.append(imu.orientation.x)
        orientation_y.append(imu.orientation.y)
        orientation_z.append(imu.orientation.z)
        orientation_w.append(imu.orientation.w)
        angular_velocity_x.append(imu.angular_velocity.x)
        angular_velocity_y.append(imu.angular_velocity.y)
        angular_velocity_z.append(imu.angular_velocity.z)
        linear_acceleration_x.append(imu.linear_acceleration.x)
        linear_acceleration_y.append(imu.linear_acceleration.y)
        linear_acceleration_z.append(imu.linear_acceleration.z)

    orientation_x = np.array(orientation_x)
    orientation_y = np.array(orientation_y)
    orientation_z = np.array(orientation_z)
    orientation_w = np.array(orientation_w)
    angular_velocity_x = np.array(angular_velocity_x)
    angular_velocity_y = np.array(angular_velocity_y)
    angular_velocity_z = np.array(angular_velocity_z)
    linear_acceleration_x = np.array(linear_acceleration_x)
    linear_acceleration_y = np.array(linear_acceleration_y)
    linear_acceleration_z = np.array(linear_acceleration_z)

    orientation_x_en = Entropy(orientation_x)
    orientation_y_en = Entropy(orientation_y)
    orientation_z_en = Entropy(orientation_z)
    orientation_w_en = Entropy(orientation_w)
    angular_velocity_x_en = Entropy(angular_velocity_x)
    angular_velocity_y_en = Entropy(angular_velocity_y)
    angular_velocity_z_en = Entropy(angular_velocity_z)
    linear_acceleration_x_en = Entropy(linear_acceleration_x)
    linear_acceleration_y_en = Entropy(linear_acceleration_y)
    linear_acceleration_z_en = Entropy(linear_acceleration_z)

    return orientation_x_en, orientation_y_en, orientation_z_en, orientation_w_en, angular_velocity_x_en, angular_velocity_y_en, angular_velocity_z_en, linear_acceleration_x_en, linear_acceleration_y_en, linear_acceleration_z_en


def compute_optimality(H_txt_path):
    ## read txt file
    # Hessian matrix = FIM
    H_mtx = np.loadtxt(H_txt_path)
    # The inverse of Hessian matrix
    # H_mtx_inv = np.loadtxt('/home/felix/panda_ws/src/franka_cal_sim/python/kalibr/H_inverse.txt')
    # Another way
    H_mtx_inv2 = np.linalg.inv(H_mtx)
    # Covariance = The inverse of FIM = The inverse of Hessian matrix
    Cov_mtx = H_mtx_inv2

    # A-Optimality
    H_Aopt = np.trace(Cov_mtx)
    print(
        'A-Optimality: minimize the trace of the covariance matrix which results in a minimization of the mean variance of the calibration parameters.'
    )
    print('H_Aopt = {}'.format(H_Aopt))

    # D-Optimality
    H_Dopt = np.linalg.det(Cov_mtx)
    print(
        'D-Optimality: minimize the determinant of the covariance matrix which results in a maximization of the differential Shannon information of the calibration parameters.'
    )
    print('H_Dopt = {}'.format(H_Dopt))

    # E-Optimality
    eig_val, eig_vec = np.linalg.eig(Cov_mtx)
    H_Eopt = max(eig_val)
    print(
        'E-Optimality: minimize the maximal eigenvalue of the covariance matrix.'
    )
    print('H_Eopt = {}'.format(H_Eopt))
    return H_Aopt, H_Dopt, H_Eopt, eig_val

def merge_bag(num_steps):
    if num_steps==1:
        rospy.wait_for_service('record_topics')
        print(num_steps)
        os.chdir(kalibr_path[:-1])
        print(kalibr_path[:-1])
        os.system("cp data_tmp.bag data.bag")
    else:
        #copy current data
        rospy.wait_for_service('record_topics')
        os.chdir(kalibr_path[:-1])
        print(kalibr_path[:-1])
        os.system("cp data.bag data_copy.bag")
        #merge bag
        with rosbag.Bag(kalibr_path + 'data.bag', 'w') as outbag:
            for topic, msg, t in rosbag.Bag(kalibr_path +
                                            'data_copy.bag').read_messages():
                outbag.write(topic, msg, t)
            for topic, msg, t in rosbag.Bag(kalibr_path +
                                            'data_tmp.bag').read_messages():
                outbag.write(topic, msg, t)

            outbag.close()

def stop_recording():
    rospy.wait_for_service('record_topics')
    stop_bag_record = rospy.ServiceProxy('stop_recording', StopRecording)
    resp1 = stop_bag_record("data_tmp.bag")

def start_recording():
    rospy.wait_for_service('record_topics')
    bag_record = rospy.ServiceProxy('record_topics', RecordTopics)
    resp1 = bag_record("data_tmp.bag",
                       ["/simple_camera/image_raw", "/imu_real"])

def wait(num):
    for i in range(num):
        i = i+1


def read_H():
    # compute observability
    H_Aopt = 0
    H_Dopt = 0
    H_Eopt = 0
    H_eig = np.zeros((6,))
    for root, dirs, files in os.walk(kalibr_path):
        if "H.txt" in files:
            H_txt_path = kalibr_path+'H.txt'
            H_Aopt, H_Dopt, H_Eopt, H_eig = compute_optimality(H_txt_path)
            os.chdir(kalibr_path[:-1])
            os.system("rm H.txt")
    #remove H file:
    return H_Aopt, H_Dopt, H_Eopt, H_eig

def try_kalibr_total():
    try:
        if os.system(k_command_total) != 0:
            raise Exception('wrongcommand does not exist')
    except:
        print("command does not work")

def try_kalibr():
    try:
        if os.system(k_command) != 0:
            raise Exception('wrongcommand does not exist')
    except:
        print("command does not work")

def printmemory(var):

    sys.getsizeof(var)


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
    global last_frame_corners
    global opts
    global ipts
    global record
    global img_cov
    global total_entropy


    rospy.loginfo(rospy.get_caller_id() + 'I heard image %s', len(image_data))
    rospy.loginfo(rospy.get_caller_id() + 'I heard imu %s', len(imu_data))
    rospy.loginfo(rospy.get_caller_id() + 'I heard state %s', len(state_data))
    record = 0


    #stop record

    if req.reset == 1:
        #stop record
        #stop_recording()

        #remove history data
        os.chdir(kalibr_path[:-1])
        os.system("rm data.bag")
        os.system("rm data_copy.bag")
        os.system("rm data_tmp.bag")
        # bag = rosbag.Bag('data_tmp.bag', 'w')
        # bag.close()


        #clear all the record data
        image_data[:]=[]
        del imu_data[:]
        del state_data[:]
        del objpoints[:]
        del imgpoints[:]
        del good_corners[:]
        del db[:]
        del opts[:]
        del ipts[:]
        obs_n = 0
        img_cov = 0
        total_entropy = 0
        last_frame_corners = None
        goodenough = False
        num_step = 0

        #feed back the update
        res = estimateSrvResponse()
        res.par_upd = [0, 0, 0, 0, 0, 0]
        res.obs = 0
        res.coverage = 0

        #start recording
        #start_recording()
        return res

    if not req.reset == 1:
        num_step += 1
        rospy.loginfo(rospy.get_caller_id() + 'start estimate')
        res = estimateSrvResponse()
        res.par_upd = [0, 0, 0, 0, 0, 0]
        res.obs = 0
        res.coverage = 0


        # estimation
        # camera intrinsic calibration

        #imgpoints, best_mtx = camera_intrinsic_calibration(req, image_data)
        best_mtx, ipts, opts, progress = camera_intrinsic_calibration2(
            req, image_data)
        rospy.loginfo(rospy.get_caller_id() + 'I get parameters %s',
                      best_mtx[0, 0])

        #merge bags
        #stop_recording()
        rospy.loginfo(rospy.get_caller_id() + 'before waiting')
        time.sleep(2)
        merge_bag(num_step)

        #cam imu calibration
        reproj_err = 1
        extrinsic = np.identity(4)
        H_Aopt = 0
        H_eig = np.zeros((6,))


        if num_step == 3:
            #call kalibr
            os.chdir(kalibr_path[:-1])
            try_kalibr_total()
            reproj_err, extrinsic = get_kalibr_results()
            extrinsic = np.asarray(extrinsic)
            H_Aopt, H_Dopt, H_Eopt, H_eig = read_H()
        #
        #
        # start = time.time()
        # if num_step<3:
        #     #get reprojection error and extrinsics
        #     os.chdir(kalibr_path[:-1])
        #     try_kalibr()
        #     reproj_err, extrinsic = get_kalibr_results()
        #     extrinsic = np.asarray(extrinsic)
        #     # compute observability
        #     H_Aopt, H_Dopt, H_Eopt, H_eig = read_H()
        # rospy.loginfo(rospy.get_caller_id() + 'time taken: '+str(time.time()-start))
        ########not calculate observability
        # else:
        #     os.system(k_command)

        ##check for leakage
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        # Prints out a summary of the large objects
        summary.print_(sum1)
        gc.collect()
        #os.system("sudo sysctl -w vm.drop_caches=3")


        rospy.loginfo(
            rospy.get_caller_id() + ' ' + 'get kalibr reproj err %s',
            reproj_err)
        rospy.loginfo(
            rospy.get_caller_id() + ' ' + 'get kalibr extrinsic %s',
            extrinsic)

        #process extrinsic
        e_flat = extrinsic[0:3,0:3].flatten()
        rotation = PyKDL.Rotation(e_flat[0],e_flat[1],e_flat[2],e_flat[3],e_flat[4],e_flat[5],e_flat[6],e_flat[7],e_flat[8])
        rpy = np.asarray(rotation.GetRPY())
        position = extrinsic[0:3,3]
        state = np.concatenate([position.reshape(3,),rpy.reshape(3,)])


        # compute entropy
        orientation_x_en, orientation_y_en, orientation_z_en, orientation_w_en, angular_velocity_x_en, angular_velocity_y_en, angular_velocity_z_en, linear_acceleration_x_en, linear_acceleration_y_en, linear_acceleration_z_en = compute_imu_entropy(
            req, imu_data)

        # # compute the coverage
        # if len(good_corners) > 0:
        res = estimateSrvResponse()

        for c, b in good_corners:
            imgpoints.append(c)  # just for coverage calculation.
        rospy.loginfo(rospy.get_caller_id() + 'I get corners %s',
                      len(ipts))

        # ####progress measures camera coverage
        # res.coverage = np.sum(progress) - cov_n
        # cov_n = np.sum(progress)
        res.obs = (len(ipts) - obs_n)/float(max([len(image_data),1.0]))
        obs_n = len(ipts)
        img_cov+=res.obs

        # res.obs = 1.0 * len(db) / 10.0 - obs_n
        # obs_n = 1.0 * len(db) / 10.0
        # get parameter update
        # compute the observation
        rospy.loginfo(rospy.get_caller_id() + 'I get image coverage %s', img_cov)
        res.obs += -H_Aopt*10000000
        res.par_upd = np.concatenate([state,np.asarray(H_eig)*10000000])
        res.coverage = (orientation_x_en+orientation_w_en+orientation_y_en+orientation_z_en)*0.1
        rospy.loginfo(rospy.get_caller_id() + 'I get par_upd %s', res.par_upd)
        rospy.loginfo(rospy.get_caller_id() + 'I get obs %s',
                      res.obs)
        rospy.loginfo(rospy.get_caller_id() + 'I get coverage %s',
                      res.coverage)
        total_entropy+=res.coverage
        rospy.loginfo(rospy.get_caller_id() + 'I get total entropy %s', total_entropy)

        # res.obs = 1.0 * len(db) / 20.0 - obs_n
        # obs_n = 1.0 * len(db) / 20.0
        # rospy.loginfo(rospy.get_caller_id() + 'I get db %s', len(db))
        # rospy.loginfo(rospy.get_caller_id() + 'I get good corners %s',
        #               len(good_corners))

        #start_recording()
        return res



def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    global kalibr_path
    rospy.init_node('listener', anonymous=True)

    kalibr_path = rospy.get_param("/rl_client/kalibr_path")

    rospy.Subscriber('/simple_camera/image_raw', Image, image_callback)
    rospy.Subscriber('/imu_real', Imu, imu_callback)
    # rospy.Subscriber('/gazebo/link_states', LinkStates, state_callback)

    record_s = rospy.Service('record_service', recordSrv, record_callback)
    estimate_s = rospy.Service('estimate_service', estimateSrv,
                               estimate_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
