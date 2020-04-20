# This is the old version! Now we don't use it anymore.

import rospy
from std_msgs.msg import String
from franka_cal_sim.srv import recordSrv, estimateSrv, recordSrvResponse, estimateSrvResponse
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
import numpy as np
import tensorflow as tf
import cv2
from cv_bridge import CvBridge
from random import sample

bridge = CvBridge()
#define data structures
image_data = []
imu_data = []
state_data = []


##subscribe topics
def image_callback(data):
    #convert ros msg to cv bridge
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


step = []
##helper functions
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

max_x = 0
max_y = 0
min_x = 0
min_y = 0
obs_n = 0


def estimate_callback(req):
    global image_data
    global state_data
    global imu_data
    global objpoints
    global imgpoints
    rospy.loginfo(rospy.get_caller_id() + 'I heard image %s', len(image_data))
    rospy.loginfo(rospy.get_caller_id() + 'I heard imu %s', len(imu_data))
    rospy.loginfo(rospy.get_caller_id() + 'I heard state %s', len(state_data))
    local_img_data = image_data
    t = 0
    if req.reset == 1:
        #clear all the record data

        global obs_n
        del image_data[:]
        del imu_data[:]
        del state_data[:]
        del objpoints[:]
        del imgpoints[:]
        max_x = 0
        max_y = 0
        min_x = 0
        min_y = 0
        obs_n = 0

        #feed back the update
        res = estimateSrvResponse()
        res.par_upd = []
        res.obs = 0
        res.coverage = 0
        return res

    if not req.reset == 1:
        if len(image_data) == 0:
            res = estimateSrvResponse()
            res.par_upd = [0, 0, 0, 0]
            res.obs = 0
            res.coverage = 0

            return res

        ##estimation
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
                    0.001)
        criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
                        0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        rospy.loginfo(rospy.get_caller_id() + 'get corner')

        for img in local_img_data:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(
                gray, (7, 6),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

            # If found, add object points, image points (after refining them)
            if ret == True and len(corners) < 50 and len(local_img_data) > 30:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (9, 9), (-1, -1),
                                            criteria)
                imgpoints.append(corners2)

        rospy.loginfo(rospy.get_caller_id() + 'compute intrinsics')

        list1 = np.arange(0, len(imgpoints), 1)
        mtx = np.zeros((3, 3))
        if len(req.params) > 3:
            mtx[0, 0] = req.params[0]
            mtx[1, 1] = req.params[1]
            mtx[0, 2] = req.params[2]
            mtx[1, 2] = req.params[3]
        mtx[2, 2] = 1

        #optimize data step by step based on sampled imgs, get best one
        min_error = 1000
        best_mtx = mtx
        if len(imgpoints) < 200 and len(local_img_data) > 30:
            for i in range(1):
                cur_data = list1
                if len(imgpoints) > 40:
                    cur_data = sample(list1, 40)
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
                    rvecs = np.asarray([1.0, 0, 0])
                    tvecs = np.asarray([0, 0, 0])
                    dist = np.asarray([0, 0, 0, 0, 0])
                    rospy.loginfo(rospy.get_caller_id() + 'No image')

            # #evaluate
            # tot_error = 0
            # mean_error = 0
            # for j in range(len(cur_obj)):
            #     imgpoints2, _ = cv2.projectPoints(cur_obj[j], rvecs[j], tvecs[j], mtx, dist)
            #     error = cv2.norm(cur_img[j],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            #     tot_error += error
            # mean_error = tot_error/max([len(cur_obj),1])
            # if mean_error<min_error:
            #     min_error = mean_error
            #     best_mtx = mtx
        best_mtx = mtx

        rospy.loginfo(rospy.get_caller_id() + 'I get corners %s',
                      len(imgpoints))
        rospy.loginfo(rospy.get_caller_id() + 'I get parameters %s', best_mtx)

        ##compute the results
        #compute the coverage
        rospy.loginfo(rospy.get_caller_id() + 'send back')
        if len(imgpoints) > 0:
            res = estimateSrvResponse()
            global max_x
            global max_y
            global min_x
            global min_y
            global obs_n
            img_flat = np.reshape(np.asarray(imgpoints), (-1, 3))
            res.coverage = np.max(img_flat, axis=0)[0] - max_x - np.min(
                img_flat, axis=0)[0] + min_x
            res.coverage = res.coverage + np.max(
                img_flat, axis=0)[1] - max_y - np.min(img_flat,
                                                      axis=0)[0] + min_y
            res.obs = 0
            if (len(imgpoints) < 100):
                res.obs = 1.0 * len(imgpoints) / len(local_img_data)
            max_x = np.max(img_flat, axis=0)[0]
            max_y = np.max(img_flat, axis=0)[1]
            min_x = np.min(img_flat, axis=0)[0]
            min_y = np.min(img_flat, axis=0)[1]
            obs_n = res.obs

            #get parameter update
            res.par_upd = [
                best_mtx[0, 0], best_mtx[1, 1], best_mtx[0, 2], best_mtx[1, 2]
            ]
            return res
        else:
            res = estimateSrvResponse()
            res.obs = 0
            res.coverage = 0
            res.par_upd = [0, 0, 0, 0]
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
