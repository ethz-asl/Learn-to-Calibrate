import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy.spatial.transform import Rotation as R
import math

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt


class CalEnv(gym.Env):
    def __init__(self, g=10.0):
        '''the basic settings of the environment
        ' image size: 640*480
        ' a square at (0, 0, 2), the size is 80 cm
        ' a camera, with intrinsics [585.7561, 585.7561, 320.5, 240.5], initial pose [0,0,0,0,0,0]
        ' camera initial coordinate system (world coordinate) x: point from camera to the board, z: up

        init state:
        0*14

        observation space 14 dim: 
        [coverage bound(8),pose(6)]
        [Umin,Umax,Vmin,Vmax,sizemin,sizemax,skewmin,skewmax,x,y,z,alpha (z),beta (y),gamma (x)]
        [0.,0.,0.,0.,0.,0.,0.,0.,-0.3,-0.2,-0.2,-0.2,-0.2,-0.2] ~ [1.,1.,1.,1.,1.,1.,1.,1.,0.3,0.2,0.2,0.2,0.2,0.2]

        action space 6 dim:
        [rho,theta,phi,alpha,beta,gamma]
        [-0.4,-1.57,-1.57,-0.4,-0.4,-0.4] ~ [0.4,1.57,1.57,0.4,0.4,0.4]
        
        '''
        #image size
        self.img_wid = 640
        self.img_hei = 480

        #half square size
        self.l = 0.8
        #square points
        # 4*4, columns are point coordinates and an additional 1.
        self.square_points = [[0.4, 0.4, 2.0], [0.4, -0.4, 2.0],
                              [-0.4, -0.4, 2.0], [-0.4, 0.4, 2.0]]
        self.square_points = np.asarray(self.square_points)
        self.square_points = self.square_points.T
        self.square_points = np.concatenate(
            [self.square_points, np.ones((1, 4))], axis=0)

        #intrinsics
        self.int = [585.7561, 585.7561, 320.5, 240.5]
        self.K = np.asarray([[self.int[0], 0., self.int[2]],
                             [0., self.int[1], self.int[3]], [0., 0., 1.]])

        #state space
        self.state_dim = 14
        self.state_high = np.asarray(
            [1., 1., 1., 1., 1., 1., 1., 1., 0.3, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.state_low = np.asarray([
            0., 0., 0., 0., 0., 0., 0., 0., -0.3, -0.2, -0.2, -0.2, -0.2, -0.2
        ])

        self.observation_space = spaces.Box(low=self.state_low,
                                            high=self.state_high,
                                            dtype=np.float32)

        #action space
        self.act_dim = 6
        self.act_high = np.asarray([0.4, 1.57, 1.57, 0.4, 0.4, 0.4])
        self.act_low = np.asarray([-0.4, -1.57, -1.57, -0.4, -0.4, -0.4])

        self.action_space = spaces.Box(low=self.act_low,
                                       high=self.act_high,
                                       dtype=np.float32)

        self.seed()

        plt.ion()
        
        fig = plt.figure(figsize=plt.figaspect(2.))
        # fig.suptitle('A tale of 2 subplots')
        self.fig = fig
        # self.ax2.set_xlim((0, 640))
        # self.ax2.set_ylim((0, 480))
        # self.ax = plt.axes(projection='3d')
        self.state_hist = []
        self.fig.show()

        h_img_uvs = np.dot(self.K, self.square_points[:3, :])
        self.img_uvs_init = np.divide(h_img_uvs[:2, :], h_img_uvs[2, :])
        self.img_square_points = self.img_uvs_init

       
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        '''
        process:
        'get current state
        'read the current pose, apply tranformation according to the action, get next pose
        'project the 4 points to the image view
        'compute new coverage
        'compute reward: sum of change of coverage of each term, minus a little path length max reward:  1.5
        
        'compute done: if each item of coverage > [0.6, 0.6, 0.05, 0.05]
        'visualize
        '''

        # get current state
        cur_pose = self.state[-6:]

        # apply transformation
        u = np.clip(u, self.act_low, self.act_high)  #clip the action
        cur_zyx = cur_pose[-3:]
        cur_rot = R.from_euler('zyx', cur_zyx)  #get rotation from eular
        rot_mat_wc = cur_rot.as_matrix()  # R_wc

        translation = np.asarray([
            u[0] * math.sin(u[2]) * math.cos(u[1]),
            u[0] * math.sin(u[2]) * math.sin(u[1]), u[0] * math.cos(u[2])
        ])
        delta_rot = R.from_euler(
            'zyx', u[-3:])  # get applied translation and rotation from action

        new_xyz = cur_pose[0:3] + translation  #compute new translation

        rot_mat_cd = delta_rot.as_matrix()
        rot_mat_wd = np.dot(rot_mat_wc, rot_mat_cd)
        new_rot = R.from_matrix(rot_mat_wd)
        new_zyx = new_rot.as_euler('zyx')  #compute new rotation

        new_pose = np.concatenate([new_xyz, new_zyx])

        new_pose = np.clip(new_pose, self.state_low[-6:],
                           self.state_high[-6:])  #clip the state
        new_rot = R.from_euler('zyx', new_pose[-3:])  #get the rot again

        # apply projection model
        rot_mat_wd = new_rot.as_matrix()
        t_vec_wd = np.asarray(new_pose[0:3].reshape((-1, 1)))

        trans_mat_wd = np.concatenate([rot_mat_wd, t_vec_wd], axis=1)
        trans_mat_wd = np.concatenate(
            [trans_mat_wd, np.asarray([[0., 0., 0., 1]])],
            axis=0)  # get transformation mat T_wc = [r_wc,t_wc;0s,1]
        trans_mat_dw = np.linalg.inv(trans_mat_wd)  # take inverse to get T_cw

        square_d = np.dot(trans_mat_dw, self.square_points)

        h_img_uvs = np.dot(self.K, square_d[:3, :])
        img_uvs = np.divide(h_img_uvs[:2, :], h_img_uvs[2, :])
        self.img_square_points = img_uvs

        # compute new coveragemin and coveragemax
        img_uv_center = np.mean(img_uvs, axis=1)

        n_center = np.divide(img_uv_center,
                             np.asarray([self.img_wid, self.img_hei
                                         ]))  # normalize the coordinate

        new_Umin = min(self.state[0], n_center[0])
        new_Umax = max(self.state[1], n_center[0])
        new_Vmin = min(self.state[2], n_center[1])
        new_Vmax = max(self.state[3], n_center[1])  # compute UV coverage

        a = (img_uvs[:, 0] - img_uvs[:, 1]).reshape((-1, 1))
        b = (img_uvs[:, 2] - img_uvs[:, 1]).reshape((-1, 1))
        area = np.linalg.det(np.concatenate(
            [a, b], axis=1))  # compute area of the square in the image
        size = area / (self.img_wid * self.img_hei
                       )  # compute portion of square in the image
        new_size_min = min(self.state[4], size)
        new_size_max = max(self.state[5], size)  #compute size coverage

        angle = math.acos(
            np.dot(a.T, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        skew = (math.pi - angle) / (math.pi)
        new_skew_min = min(self.state[6], skew)
        new_skew_max = max(self.state[7], skew)

        # compute reward
        reward = new_Umax - self.state[1] + self.state[0] - new_Umin
        reward += new_Vmax - self.state[3] + self.state[2] - new_Vmin
        reward += new_size_max - self.state[5] + self.state[4] - new_size_min
        reward += new_skew_max - self.state[7] + self.state[
            6] - new_skew_min  # add coverage

        #TODO: modified here
        reward -= abs(u[0]) * 0.1
        reward -= np.linalg.norm(u[-3:]) * 0.02  # minus the length

        reward *= 100

        # compute done
        done = (new_Umax - new_Umin) > 0.6 and (new_Vmax - new_Vmin) > 0.6
        done = done and (new_skew_max - new_skew_min) > 0.05 and (
            new_size_max - new_size_min) > 0.05

        if done:
            reward += 50

        self.state = np.asarray([
            new_Umin,
            new_Umax,
            new_Vmin,
            new_Vmax,
            new_size_min,
            new_size_max,
            new_skew_min,
            new_skew_max,
        ])
        self.state = np.concatenate([self.state, new_pose])

        return self.state, reward, done, {
            'state: ': self.state,
            'action: ': u,
            'img_uv_center': img_uv_center
        }

        

    def reset(self):
        '''
        reset: return to the origin, clear the coverage
        state = [0.5,0.5,0.5,0.5,size,size,0.5,0.5]
        compute initial area
        '''

        self.state = np.concatenate([np.ones((8, )) * 0.5, np.zeros((6, ))])

        h_img_uvs = np.dot(self.K, self.square_points[:3, :])
        img_uvs = np.divide(h_img_uvs[:2, :], h_img_uvs[2, :])

        a = (img_uvs[:, 0] - img_uvs[:, 1]).reshape((-1, 1))
        b = (img_uvs[:, 2] - img_uvs[:, 1]).reshape((-1, 1))
        area = np.linalg.det(np.concatenate(
            [a, b], axis=1))  # compute area of the square in the image
        size = area / (self.img_wid * self.img_hei)

        self.state[4] = size
        self.state[5] = size

        #initialize visualization
        self.state_hist = []
        self.img_square_points = self.img_uvs_init


        return self.state

    def visualize(self):
        '''
        visualize the current state:
        '3D visualize of the camera pose and the square
        '2D image visualize
        '''
        if len(self.state_hist) == 0:
            # plt.show(block=True)
            plt.clf()
            self.ax = self.fig.add_subplot(2, 1, 1, projection='3d')
            self.ax2 = self.fig.add_subplot(2, 1, 2)
            

        #visualize the square

        square = np.concatenate(
            [self.square_points, self.square_points[:, 0].reshape((-1, 1))],
            axis=1)
        # print(square)
        self.ax.plot3D(square[0, :], square[1, :], square[2, :], 'gray')

        #visualize the camera
        cur_pose = self.state[-6:]

        self.ax.scatter3D(cur_pose[0],
                          cur_pose[1],
                          cur_pose[2],
                          'o',
                          color='green')  # plot position

        c_origin = cur_pose[:3]
        cur_zyx = cur_pose[-3:]
        cur_rot = R.from_euler('zyx', cur_zyx)  #get rotation from eular
        rot_mat_wc = cur_rot.as_matrix()  # R_wc
        z_one = np.dot(rot_mat_wc, np.asarray([0., 0., 0.2])) + c_origin

        direction = np.concatenate(
            [c_origin.reshape(
                (3, 1)), z_one.reshape((3, 1))], axis=1)

        self.ax.plot3D(direction[0, :], direction[1, :], direction[2, :],
                       'red')  #plot direction

        # visualize the trajectory
        self.state_hist.append(self.state.reshape((-1, 1)))
        hist_array = np.concatenate(self.state_hist, axis=1)
        self.ax.plot3D(hist_array[8, :], hist_array[9, :], hist_array[10, :],
                       'black')

        self.ax.set_ylim((-0.5, 0.5))
        self.ax.set_xlim((-0.5, 0.5))
        self.ax.set_zlim((-0.5, 2))
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        #visualize the 2d points
        x = [[self.img_square_points[0, 0], self.img_square_points[0, 1]],
             [self.img_square_points[0, 1], self.img_square_points[0, 2]],
             [self.img_square_points[0, 2], self.img_square_points[0, 3]],
             [self.img_square_points[0, 3], self.img_square_points[0, 0]]]
        y = [[self.img_square_points[1, 0], self.img_square_points[1, 1]],
             [self.img_square_points[1, 1], self.img_square_points[1, 2]],
             [self.img_square_points[1, 2], self.img_square_points[1, 3]],
             [self.img_square_points[1, 3], self.img_square_points[1, 0]]]

        for i in range(len(x)):
            self.ax2.plot(x[i], y[i], color='r')
            self.ax2.scatter(x[i], y[i], color='b')

        self.ax2.xaxis.tick_top()
        self.ax2.yaxis.tick_left()
        self.ax2.set_xlim((0, 640))
        self.ax2.set_ylim((480, 0))

        plt.pause(1)




class DoubleCalEnv(gym.Env):
    def __init__(self, g=10.0):
        '''the basic settings of the environment
        ' image size: 640*480
        ' a square at (0, 0, 2), the size is 80 cm
        ' a camera, with intrinsics [585.7561, 585.7561, 320.5, 240.5], initial pose [0,0,0,0,0,0]
        ' camera initial coordinate system (world coordinate) x: point from camera to the board, z: up

        init state:
        0*14

        observation space 18 dim: 
        [coverage bound(12),pose(6)]
        [U00min, V00min, U01min, V01max, U11max, V11max, U10max, V10min, sizemin,sizemax,skewmin,skewmax,x,y,z,alpha (z),beta (y),gamma (x)]
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.3,-0.2,-0.2,-0.2,-0.2,-0.2] ~ [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.3,0.2,0.2,0.2,0.2,0.2]

        action space 6 dim:
        [rho,theta,phi,alpha,beta,gamma]
        [-0.4,-1.57,-1.57,-0.4,-0.4,-0.4] ~ [0.4,1.57,1.57,0.4,0.4,0.4]
        
        '''
        #image size
        self.img_wid = 640
        self.img_hei = 480

        #half square size
        self.l = 0.8
        #square points
        # 4*4, columns are point coordinates and an additional 1.
        self.square_points = [[0.4, 0.4, 2.0], [0.4, -0.4, 2.0],
                              [-0.4, -0.4, 2.0], [-0.4, 0.4, 2.0]]
        self.square_points = np.asarray(self.square_points)
        self.square_points = self.square_points.T
        self.square_points = np.concatenate(
            [self.square_points, np.ones((1, 4))], axis=0)

        #intrinsics
        self.int = [585.7561, 585.7561, 320.5, 240.5]
        self.K = np.asarray([[self.int[0], 0., self.int[2]],
                             [0., self.int[1], self.int[3]], [0., 0., 1.]])

        #state space
        self.state_dim = 18
        self.state_high = np.asarray(
            [1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1., 0.3, 0.2, 0.2, 0.3, 0.2, 0.2])
        self.state_low = np.asarray([
            0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., -0.3, -0.2, -0.2, -0.3, -0.2, -0.2
        ])

        self.observation_space = spaces.Box(low=self.state_low,
                                            high=self.state_high,
                                            dtype=np.float32)

        #action space
        self.act_dim = 6
        self.act_high = np.asarray([0.4, 1.57, 1.57, 0.6, 0.4, 0.4])
        self.act_low = np.asarray([-0.4, -1.57, -1.57, -0.6, -0.4, -0.4])

        self.action_space = spaces.Box(low=self.act_low,
                                       high=self.act_high,
                                       dtype=np.float32)

        self.seed()

        plt.ion()
        
        fig = plt.figure(figsize=plt.figaspect(2.))
        # fig.suptitle('A tale of 2 subplots')
        self.fig = fig
        # self.ax2.set_xlim((0, 640))
        # self.ax2.set_ylim((0, 480))
        # self.ax = plt.axes(projection='3d')
        self.state_hist = []
        self.fig.show()

        h_img_uvs = np.dot(self.K, self.square_points[:3, :])
        self.img_uvs_init = np.divide(h_img_uvs[:2, :], h_img_uvs[2, :])
        self.img_square_points = self.img_uvs_init

       
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        '''
        process:
        'get current state
        'read the current pose, apply tranformation according to the action, get next pose
        'project the 4 points to the image view
        'compute new coverage
        'compute reward: sum of change of coverage of each term, minus a little path length max reward:  1.5
        
        'compute done: if each item of coverage > [0.6, 0.6, 0.05, 0.05]
        'visualize
        '''

        # get current state
        cur_pose = self.state[-6:]

        # apply transformation
        u = np.clip(u, self.act_low, self.act_high)  #clip the action
        cur_zyx = cur_pose[-3:]
        cur_rot = R.from_euler('zyx', cur_zyx)  #get rotation from eular
        rot_mat_wc = cur_rot.as_matrix()  # R_wc

        translation = np.asarray([
            u[0] * math.sin(u[2]) * math.cos(u[1]),
            u[0] * math.sin(u[2]) * math.sin(u[1]), u[0] * math.cos(u[2])
        ])
        delta_rot = R.from_euler(
            'zyx', u[-3:])  # get applied translation and rotation from action

        new_xyz = cur_pose[0:3] + translation  #compute new translation

        rot_mat_cd = delta_rot.as_matrix()
        rot_mat_wd = np.dot(rot_mat_wc, rot_mat_cd)
        new_rot = R.from_matrix(rot_mat_wd)
        new_zyx = new_rot.as_euler('zyx')  #compute new rotation

        new_pose = np.concatenate([new_xyz, new_zyx])

        new_pose = np.clip(new_pose, self.state_low[-6:],
                           self.state_high[-6:])  #clip the state
        new_rot = R.from_euler('zyx', new_pose[-3:])  #get the rot again

        # apply projection model
        rot_mat_wd = new_rot.as_matrix()
        t_vec_wd = np.asarray(new_pose[0:3].reshape((-1, 1)))

        trans_mat_wd = np.concatenate([rot_mat_wd, t_vec_wd], axis=1)
        trans_mat_wd = np.concatenate(
            [trans_mat_wd, np.asarray([[0., 0., 0., 1]])],
            axis=0)  # get transformation mat T_wc = [r_wc,t_wc;0s,1]
        trans_mat_dw = np.linalg.inv(trans_mat_wd)  # take inverse to get T_cw

        square_d = np.dot(trans_mat_dw, self.square_points)

        h_img_uvs = np.dot(self.K, square_d[:3, :])
        img_uvs = np.divide(h_img_uvs[:2, :], h_img_uvs[2, :])
        self.img_square_points = img_uvs

        ## compute new coveragemin and coveragemax
        img_uv_center = np.mean(img_uvs, axis=1)

        n_center = np.divide(img_uv_center,
                             np.asarray([self.img_wid, self.img_hei
                                         ]))  # normalize the coordinate

        #TODO: modify x, y coverage
        '''
        [U00min, V00min, U01min, V01max, U11max, V11max, U10max, V10min, sizemin,sizemax,skewmin,skewmax,x,y,z,alpha (z),beta (y),gamma (x)]
        '''

        new_U00min = self.state[0]
        new_V00min = self.state[1]
        new_U01min = self.state[2]
        new_V01max = self.state[3]
        new_U11max = self.state[4]
        new_V11max = self.state[5]
        new_U10max = self.state[6]
        new_V10min = self.state[7]

        if n_center[0] < 0.5 and n_center[1] < 0.5: # 00 part
            new_U00min = min(new_U00min, n_center[0])
            new_V00min = min(new_V00min, n_center[1])

        if n_center[0] < 0.5 and n_center[1] > 0.5: # 01 part
            new_U01min = min(new_U01min, n_center[0])
            new_V01max = max(new_V01max, n_center[1])

        if n_center[0] > 0.5 and n_center[1] > 0.5: # 11 part
            new_U11max = max(new_U11max, n_center[0])
            new_V11max = max(new_V11max, n_center[1])

        if n_center[0] > 0.5 and n_center[1] < 0.5: # 10 part
            new_U10max = max(new_U10max, n_center[0])
            new_V10min = min(new_V10min, n_center[1])
        

        # area
        a = (img_uvs[:, 0] - img_uvs[:, 1]).reshape((-1, 1))
        b = (img_uvs[:, 2] - img_uvs[:, 1]).reshape((-1, 1))
        area = np.linalg.det(np.concatenate(
            [a, b], axis=1))  # compute area of the square in the image
        size = area / (self.img_wid * self.img_hei
                       )  # compute portion of square in the image
        new_size_min = min(self.state[8], size)
        new_size_max = max(self.state[9], size)  #compute size coverage

        # skew
        angle = math.acos(
            np.dot(a.T, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        skew = (math.pi - angle) / (math.pi)
        new_skew_min = min(self.state[10], skew)
        new_skew_max = max(self.state[11], skew)

        ## compute reward
        reward = self.state[0] - new_U00min + self.state[1] - new_V00min
        reward += self.state[2] - new_U01min + new_V01max - self.state[3]
        reward += new_U11max - self.state[4] + new_V11max - self.state[5]
        reward += new_U10max - self.state[6] + self.state[7] - new_V10min
        reward += new_size_max - self.state[9] + self.state[8] - new_size_min
        reward += new_skew_max - self.state[11] + self.state[10] - new_skew_min  # add coverage

        
        reward -= abs(u[0]) * 0.1
        reward -= np.linalg.norm(u[-3:]) * 0.02  # minus the length

        reward *= 100

        # compute done
        done = new_U00min < 0.4 and new_V00min < 0.4
        done = done and new_U01min < 0.4 and new_V01max > 0.6
        done = done and new_U11max > 0.6 and new_V11max > 0.6
        done = done and new_U10max > 0.6 and new_V10min < 0.4
        done = done and (new_skew_max - new_skew_min) > 0.05 and (
            new_size_max - new_size_min) > 0.05

        if done:
            reward += 50

        self.state = np.asarray([
            new_U00min,
            new_V00min,
            new_U01min,
            new_V01max,
            new_U11max,
            new_V11max,
            new_U10max,
            new_V10min,
            new_size_min,
            new_size_max,
            new_skew_min,
            new_skew_max,
        ])
        self.state = np.concatenate([self.state, new_pose])

        return self.state, reward, done, {
            'state: ': self.state,
            'action: ': u,
            'img_uv_center': img_uv_center
        }

        

    def reset(self):
        '''
        reset: return to the origin, clear the coverage
        state = [0.5,0.5,0.5,0.5,size,size,0.5,0.5]
        compute initial area
        '''

        self.state = np.concatenate([np.ones((12, )) * 0.5, np.zeros((6, ))])

        h_img_uvs = np.dot(self.K, self.square_points[:3, :])
        img_uvs = np.divide(h_img_uvs[:2, :], h_img_uvs[2, :])

        a = (img_uvs[:, 0] - img_uvs[:, 1]).reshape((-1, 1))
        b = (img_uvs[:, 2] - img_uvs[:, 1]).reshape((-1, 1))
        area = np.linalg.det(np.concatenate(
            [a, b], axis=1))  # compute area of the square in the image
        size = area / (self.img_wid * self.img_hei)

        self.state[8] = size
        self.state[9] = size

        #initialize visualization
        self.state_hist = []
        self.img_square_points = self.img_uvs_init


        return self.state

    def visualize(self):
        '''
        visualize the current state:
        '3D visualize of the camera pose and the square
        '2D image visualize
        '''
        if len(self.state_hist) == 0:
            # plt.show(block=True)
            plt.clf()
            self.ax = self.fig.add_subplot(2, 1, 1, projection='3d')
            self.ax2 = self.fig.add_subplot(2, 1, 2)
            

        #visualize the square

        square = np.concatenate(
            [self.square_points, self.square_points[:, 0].reshape((-1, 1))],
            axis=1)
        # print(square)
        self.ax.plot3D(square[0, :], square[1, :], square[2, :], 'gray')

        #visualize the camera
        cur_pose = self.state[-6:]

        self.ax.scatter3D(cur_pose[0],
                          cur_pose[1],
                          cur_pose[2],
                          'o',
                          color='green')  # plot position

        c_origin = cur_pose[:3]
        cur_zyx = cur_pose[-3:]
        cur_rot = R.from_euler('zyx', cur_zyx)  #get rotation from eular
        rot_mat_wc = cur_rot.as_matrix()  # R_wc
        z_one = np.dot(rot_mat_wc, np.asarray([0., 0., 0.2])) + c_origin

        direction = np.concatenate(
            [c_origin.reshape(
                (3, 1)), z_one.reshape((3, 1))], axis=1)

        self.ax.plot3D(direction[0, :], direction[1, :], direction[2, :],
                       'red')  #plot direction

        # visualize the trajectory
        self.state_hist.append(self.state.reshape((-1, 1)))
        hist_array = np.concatenate(self.state_hist, axis=1)
        self.ax.plot3D(hist_array[12, :], hist_array[13, :], hist_array[14, :],
                       'black')

        self.ax.set_ylim((-0.5, 0.5))
        self.ax.set_xlim((-0.5, 0.5))
        self.ax.set_zlim((-0.5, 2))
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        #visualize the 2d points
        x = [[self.img_square_points[0, 0], self.img_square_points[0, 1]],
             [self.img_square_points[0, 1], self.img_square_points[0, 2]],
             [self.img_square_points[0, 2], self.img_square_points[0, 3]],
             [self.img_square_points[0, 3], self.img_square_points[0, 0]]]
        y = [[self.img_square_points[1, 0], self.img_square_points[1, 1]],
             [self.img_square_points[1, 1], self.img_square_points[1, 2]],
             [self.img_square_points[1, 2], self.img_square_points[1, 3]],
             [self.img_square_points[1, 3], self.img_square_points[1, 0]]]

        for i in range(len(x)):
            self.ax2.plot(x[i], y[i], color='r')
            self.ax2.scatter(x[i], y[i], color='b')

        self.ax2.xaxis.tick_top()
        self.ax2.yaxis.tick_left()
        self.ax2.set_xlim((0, 640))
        self.ax2.set_ylim((480, 0))

        plt.pause(0.5)


        
class DoubleCalImuEnv(gym.Env):
    def __init__(self, g=10.0):
        '''the basic settings of the environment
        ' image size: 640*480
        ' a square at (0, 0, 2), the size is 80 cm
        ' a camera, with intrinsics [585.7561, 585.7561, 320.5, 240.5], initial pose [0,0,0,0,0,0]
        ' camera initial coordinate system (world coordinate) x: point from camera to the board, z: up

        init state:
        0*14

        observation space 18 dim: 
        [coverage bound(12),pose(6)]
        [U00min, V00min, U01min, V01max, U11max, V11max, U10max, V10min, sizemin,sizemax,skewmin,skewmax,
        dx_max, dy_max, dz_max, dalpha_max, dbeta_max, dgamma_max,
        x,y,z,alpha (z),beta (y),gamma (x)]
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
        0.,0.,0.,0.,0.,0.,
        -0.3,-0.2,-0.2,-0.2,-0.2,-0.2] ~ 
        [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
        1.,1.,1.,1.,1.,1.,
        0.3,0.2,0.2,0.2,0.2,0.2]

        action space 6 dim:
        [rho,theta,phi,alpha,beta,gamma]
        [-0.4,-1.57,-1.57,-0.4,-0.4,-0.4] ~ [0.4,1.57,1.57,0.4,0.4,0.4]
        
        '''
        #image size
        self.img_wid = 640
        self.img_hei = 480

        #half square size
        self.l = 0.8
        #square points
        # 4*4, columns are point coordinates and an additional 1.
        self.square_points = [[0.4, 0.4, 2.0], [0.4, -0.4, 2.0],
                              [-0.4, -0.4, 2.0], [-0.4, 0.4, 2.0]]
        self.square_points = np.asarray(self.square_points)
        self.square_points = self.square_points.T
        self.square_points = np.concatenate(
            [self.square_points, np.ones((1, 4))], axis=0)

        #intrinsics
        self.int = [585.7561, 585.7561, 320.5, 240.5]
        self.K = np.asarray([[self.int[0], 0., self.int[2]],
                             [0., self.int[1], self.int[3]], [0., 0., 1.]])

        #state space
        self.state_dim = 24
        self.state_high = np.asarray(
            [1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1.,1.,1.,1.,1.,1.,1., 0.3, 0.2, 0.2, 0.3, 0.2, 0.2])
        self.state_low = np.asarray([
            0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.3, -0.2, -0.2, -0.3, -0.2, -0.2
        ])

        self.observation_space = spaces.Box(low=self.state_low,
                                            high=self.state_high,
                                            dtype=np.float32)

        #action space
        self.act_dim = 6
        self.act_high = np.asarray([0.4, 1.57, 1.57, 0.6, 0.4, 0.4])
        self.act_low = np.asarray([-0.4, -1.57, -1.57, -0.6, -0.4, -0.4])

        self.action_space = spaces.Box(low=self.act_low,
                                       high=self.act_high,
                                       dtype=np.float32)

        self.seed()

        plt.ion()
        
        fig = plt.figure(figsize=plt.figaspect(2.))
        # fig.suptitle('A tale of 2 subplots')
        self.fig = fig
        # self.ax2.set_xlim((0, 640))
        # self.ax2.set_ylim((0, 480))
        # self.ax = plt.axes(projection='3d')
        self.state_hist = []
        self.fig.show()

        h_img_uvs = np.dot(self.K, self.square_points[:3, :])
        self.img_uvs_init = np.divide(h_img_uvs[:2, :], h_img_uvs[2, :])
        self.img_square_points = self.img_uvs_init

       
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        '''
        process:
        'get current state
        'read the current pose, apply tranformation according to the action, get next pose
        'project the 4 points to the image view
        'compute new coverage
        'compute reward: sum of change of coverage of each term, minus a little path length max reward:  1.5
        
        'compute done: if each item of coverage > [0.6, 0.6, 0.05, 0.05]
        'visualize
        '''

        # get current state
        cur_pose = self.state[-6:]

        # apply transformation
        u = np.clip(u, self.act_low, self.act_high)  #clip the action
        cur_zyx = cur_pose[-3:]
        cur_rot = R.from_euler('zyx', cur_zyx)  #get rotation from eular
        rot_mat_wc = cur_rot.as_matrix()  # R_wc

        translation = np.asarray([
            u[0] * math.sin(u[2]) * math.cos(u[1]),
            u[0] * math.sin(u[2]) * math.sin(u[1]), u[0] * math.cos(u[2])
        ])
        delta_rot = R.from_euler(
            'zyx', u[-3:])  # get applied translation and rotation from action

        new_xyz = cur_pose[0:3] + translation  #compute new translation

        rot_mat_cd = delta_rot.as_matrix()
        rot_mat_wd = np.dot(rot_mat_wc, rot_mat_cd)
        new_rot = R.from_matrix(rot_mat_wd)
        new_zyx = new_rot.as_euler('zyx')  #compute new rotation

        new_pose = np.concatenate([new_xyz, new_zyx])

        new_pose = np.clip(new_pose, self.state_low[-6:],
                           self.state_high[-6:])  #clip the state
        new_rot = R.from_euler('zyx', new_pose[-3:])  #get the rot again

        # apply projection model
        rot_mat_wd = new_rot.as_matrix()
        t_vec_wd = np.asarray(new_pose[0:3].reshape((-1, 1)))

        trans_mat_wd = np.concatenate([rot_mat_wd, t_vec_wd], axis=1)
        trans_mat_wd = np.concatenate(
            [trans_mat_wd, np.asarray([[0., 0., 0., 1]])],
            axis=0)  # get transformation mat T_wc = [r_wc,t_wc;0s,1]
        trans_mat_dw = np.linalg.inv(trans_mat_wd)  # take inverse to get T_cw

        square_d = np.dot(trans_mat_dw, self.square_points)

        h_img_uvs = np.dot(self.K, square_d[:3, :])
        img_uvs = np.divide(h_img_uvs[:2, :], h_img_uvs[2, :])
        self.img_square_points = img_uvs

        ## compute new coveragemin and coveragemax
        img_uv_center = np.mean(img_uvs, axis=1)

        n_center = np.divide(img_uv_center,
                             np.asarray([self.img_wid, self.img_hei
                                         ]))  # normalize the coordinate

        #TODO: modify x, y coverage
        '''
        [U00min, V00min, U01min, V01max, U11max, V11max, U10max, V10min]
        '''

        new_U00min = self.state[0]
        new_V00min = self.state[1]
        new_U01min = self.state[2]
        new_V01max = self.state[3]
        new_U11max = self.state[4]
        new_V11max = self.state[5]
        new_U10max = self.state[6]
        new_V10min = self.state[7]

        if n_center[0] < 0.5 and n_center[1] < 0.5: # 00 part
            new_U00min = min(new_U00min, n_center[0])
            new_V00min = min(new_V00min, n_center[1])

        if n_center[0] < 0.5 and n_center[1] > 0.5: # 01 part
            new_U01min = min(new_U01min, n_center[0])
            new_V01max = max(new_V01max, n_center[1])

        if n_center[0] > 0.5 and n_center[1] > 0.5: # 11 part
            new_U11max = max(new_U11max, n_center[0])
            new_V11max = max(new_V11max, n_center[1])

        if n_center[0] > 0.5 and n_center[1] < 0.5: # 10 part
            new_U10max = max(new_U10max, n_center[0])
            new_V10min = min(new_V10min, n_center[1])

        #TODO: modify coverage for IMU
        '''
        dx_max, dy_max, dz_max, dalpha_max, dbeta_max, dgamma_max
        '''
        new_pose_max = self.state[12:18]

        diff_pose = np.divide(np.abs(new_pose - cur_pose), self.state_high[-6:] - self.state_low[-6:])

        comp = np.concatenate([new_pose_max.reshape((-1,1)), 
                                diff_pose.reshape((-1,1))],
                                axis=1)

        new_pose_max = np.max(comp, axis = 1)

        
        # area
        a = (img_uvs[:, 0] - img_uvs[:, 1]).reshape((-1, 1))
        b = (img_uvs[:, 2] - img_uvs[:, 1]).reshape((-1, 1))
        area = np.linalg.det(np.concatenate(
            [a, b], axis=1))  # compute area of the square in the image
        size = area / (self.img_wid * self.img_hei
                       )  # compute portion of square in the image
        new_size_min = min(self.state[8], size)
        new_size_max = max(self.state[9], size)  #compute size coverage

        # skew
        angle = math.acos(
            np.dot(a.T, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        skew = (math.pi - angle) / (math.pi)
        new_skew_min = min(self.state[10], skew)
        new_skew_max = max(self.state[11], skew)

        ## compute reward
        reward = self.state[0] - new_U00min + self.state[1] - new_V00min
        reward += self.state[2] - new_U01min + new_V01max - self.state[3]
        reward += new_U11max - self.state[4] + new_V11max - self.state[5]
        reward += new_U10max - self.state[6] + self.state[7] - new_V10min
        reward += new_size_max - self.state[9] + self.state[8] - new_size_min
        reward += new_skew_max - self.state[11] + self.state[10] - new_skew_min  # add coverage

        #TODO: add pose coverage
        reward += reward * 0.5 + np.sum(new_pose_max - self.state[12:18])

        
        reward -= abs(u[0]) * 0.1
        reward -= np.linalg.norm(u[-3:]) * 0.02  # minus the length

        reward *= 100

        # compute done
        done = new_U00min < 0.45 and new_V00min < 0.45
        done = done and new_U01min < 0.45 and new_V01max > 0.55
        done = done and new_U11max > 0.55 and new_V11max > 0.55
        done = done and new_U10max > 0.55 and new_V10min < 0.45
        done = done and (new_skew_max - new_skew_min) > 0.05 and (
            new_size_max - new_size_min) > 0.05
        done = done and (new_pose_max > 0.3).all()

        if done:
            reward += 200

        self.state = np.asarray([
            new_U00min,
            new_V00min,
            new_U01min,
            new_V01max,
            new_U11max,
            new_V11max,
            new_U10max,
            new_V10min,
            new_size_min,
            new_size_max,
            new_skew_min,
            new_skew_max,
        ])

        #TODO: add pose coverage
        self.state = np.concatenate([self.state, new_pose_max.reshape((-1,))])
        self.state = np.concatenate([self.state, new_pose])

        return self.state, reward, done, {
            'state: ': self.state,
            'action: ': u,
            'img_uv_center': img_uv_center
        }

        

    def reset(self):
        '''
        reset: return to the origin, clear the coverage
        state = [0.5,0.5,0.5,0.5,size,size,0.5,0.5]
        compute initial area
        '''

        self.state = np.concatenate([np.ones((12, )) * 0.5, np.zeros((12, ))])

        h_img_uvs = np.dot(self.K, self.square_points[:3, :])
        img_uvs = np.divide(h_img_uvs[:2, :], h_img_uvs[2, :])

        a = (img_uvs[:, 0] - img_uvs[:, 1]).reshape((-1, 1))
        b = (img_uvs[:, 2] - img_uvs[:, 1]).reshape((-1, 1))
        area = np.linalg.det(np.concatenate(
            [a, b], axis=1))  # compute area of the square in the image
        size = area / (self.img_wid * self.img_hei)

        self.state[8] = size
        self.state[9] = size

        #initialize visualization
        self.state_hist = []
        self.img_square_points = self.img_uvs_init


        return self.state

    def visualize(self):
        '''
        visualize the current state:
        '3D visualize of the camera pose and the square
        '2D image visualize
        '''
        if len(self.state_hist) == 0:
            # plt.show(block=True)
            plt.clf()
            self.ax = self.fig.add_subplot(2, 1, 1, projection='3d')
            self.ax2 = self.fig.add_subplot(2, 1, 2)
            

        #visualize the square

        square = np.concatenate(
            [self.square_points, self.square_points[:, 0].reshape((-1, 1))],
            axis=1)
        # print(square)
        self.ax.plot3D(square[0, :], square[1, :], square[2, :], 'gray')

        #visualize the camera
        cur_pose = self.state[-6:]

        self.ax.scatter3D(cur_pose[0],
                          cur_pose[1],
                          cur_pose[2],
                          'o',
                          color='green')  # plot position

        c_origin = cur_pose[:3]
        cur_zyx = cur_pose[-3:]
        cur_rot = R.from_euler('zyx', cur_zyx)  #get rotation from eular
        rot_mat_wc = cur_rot.as_matrix()  # R_wc
        z_one = np.dot(rot_mat_wc, np.asarray([0., 0., 0.2])) + c_origin

        direction = np.concatenate(
            [c_origin.reshape(
                (3, 1)), z_one.reshape((3, 1))], axis=1)

        self.ax.plot3D(direction[0, :], direction[1, :], direction[2, :],
                       'red')  #plot direction

        # visualize the trajectory
        self.state_hist.append(self.state.reshape((-1, 1)))
        hist_array = np.concatenate(self.state_hist, axis=1)
        self.ax.plot3D(hist_array[-6, :], hist_array[-5, :], hist_array[-4, :],
                       'black')

        self.ax.set_ylim((-0.5, 0.5))
        self.ax.set_xlim((-0.5, 0.5))
        self.ax.set_zlim((-0.5, 2))
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        #visualize the 2d points
        x = [[self.img_square_points[0, 0], self.img_square_points[0, 1]],
             [self.img_square_points[0, 1], self.img_square_points[0, 2]],
             [self.img_square_points[0, 2], self.img_square_points[0, 3]],
             [self.img_square_points[0, 3], self.img_square_points[0, 0]]]
        y = [[self.img_square_points[1, 0], self.img_square_points[1, 1]],
             [self.img_square_points[1, 1], self.img_square_points[1, 2]],
             [self.img_square_points[1, 2], self.img_square_points[1, 3]],
             [self.img_square_points[1, 3], self.img_square_points[1, 0]]]

        for i in range(len(x)):
            self.ax2.plot(x[i], y[i], color='r')
            self.ax2.scatter(x[i], y[i], color='b')

        self.ax2.xaxis.tick_top()
        self.ax2.yaxis.tick_left()
        self.ax2.set_xlim((0, 640))
        self.ax2.set_ylim((480, 0))

        plt.pause(0.5)



class DoubleCalImuSetEnv(gym.Env):
    def __init__(self, g=10.0):
        '''the basic settings of the environment
        ' image size: 640*480
        ' a square at (0, 0, 2), the size is 80 cm
        ' a camera, with intrinsics [585.7561, 585.7561, 320.5, 240.5], initial pose [0,0,0,0,0,0]
        ' camera initial coordinate system (world coordinate) x: point from camera to the board, z: up

        init state:
        0*14

        observation space 18 dim: 
        [coverage bound(12),pose(6)]
        [U00min, V00min, U01min, V01max, U11max, V11max, U10max, V10min, sizemin,sizemax,skewmin,skewmax,
        dx_max, dy_max, dz_max, dalpha_max, dbeta_max, dgamma_max,
        x,y,z,alpha (z),beta (y),gamma (x)]
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
        0.,0.,0.,0.,0.,0.,
        -0.3,-0.2,-0.2,-0.2,-0.2,-0.2] ~ 
        [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
        1.,1.,1.,1.,1.,1.,
        0.3,0.2,0.2,0.2,0.2,0.2]

        action space 6 dim:
        [rho,theta,phi,alpha,beta,gamma]
        [-0.4,-1.57,-1.57,-0.4,-0.4,-0.4] ~ [0.4,1.57,1.57,0.4,0.4,0.4]
        
        '''
        #image size
        self.img_wid = 640
        self.img_hei = 480

        #half square size
        self.l = 0.8
        #square points
        # 4*4, columns are point coordinates and an additional 1.
        self.square_points = [[0.4, 0.4, 2.0], [0.4, -0.4, 2.0],
                              [-0.4, -0.4, 2.0], [-0.4, 0.4, 2.0]]
        self.square_points = np.asarray(self.square_points)
        self.square_points = self.square_points.T
        self.square_points = np.concatenate(
            [self.square_points, np.ones((1, 4))], axis=0)

        #intrinsics
        self.int = [585.7561, 585.7561, 320.5, 240.5]
        self.K = np.asarray([[self.int[0], 0., self.int[2]],
                             [0., self.int[1], self.int[3]], [0., 0., 1.]])

        #state space
        #self.set_obs_dim = 16
        self.set_obs_dim = 16
        self.nom_obs_dim = 6

        self.set_state = []
        self.nom_state = np.zeros((6,))

        self.nom_state_high = np.asarray([0.3, 0.2, 0.2, 0.3, 0.2, 0.2])
        self.nom_state_low = -self.nom_state_high

        self.state_dim = 24
        self.state_high = np.asarray(
            [1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1.,1.,1.,1.,1.,1.,1., 0.3, 0.2, 0.2, 0.3, 0.2, 0.2])
        self.state_low = np.asarray([
            0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.3, -0.2, -0.2, -0.3, -0.2, -0.2
        ])

        self.observation_space = spaces.Box(low=self.state_low,
                                            high=self.state_high,
                                            dtype=np.float32)

        #action space
        self.act_dim = 6
        self.act_high = np.asarray([0.4, 1.57, 1.57, 0.6, 0.4, 0.4])
        self.act_low = np.asarray([-0.4, -1.57, -1.57, -0.6, -0.4, -0.4])

        self.action_space = spaces.Box(low=self.act_low,
                                       high=self.act_high,
                                       dtype=np.float32)

        self.seed()

        plt.ion()
        
        fig = plt.figure(figsize=plt.figaspect(2.))
        # fig.suptitle('A tale of 2 subplots')
        self.fig = fig
        # self.ax2.set_xlim((0, 640))
        # self.ax2.set_ylim((0, 480))
        # self.ax = plt.axes(projection='3d')
        self.state_hist = []
        self.fig.show()

        h_img_uvs = np.dot(self.K, self.square_points[:3, :])
        self.img_uvs_init = np.divide(h_img_uvs[:2, :], h_img_uvs[2, :])
        self.img_square_points = self.img_uvs_init
        n_uvs = self.img_uvs_init / np.asarray([self.img_wid, self.img_hei
                                         ]).reshape((2, 1))
        obs = np.concatenate([n_uvs.reshape((-1,)), 0.5*np.ones((2,)), self.nom_state])
        #obs = np.concatenate([0.5*np.ones((2,)), n_uvs, self.nom_state])
        self.set_state = np.repeat(obs.reshape((-1, 1)), 11, axis=1)
        self.count = 1

       
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        '''
        process:
        'get current state
        'read the current pose, apply tranformation according to the action, get next pose
        'project the 4 points to the image view
        'compute new coverage
        'compute reward: sum of change of coverage of each term, minus a little path length max reward:  1.5
        
        'compute done: if each item of coverage > [0.6, 0.6, 0.05, 0.05]
        'visualize
        '''

        # get current state
        cur_pose = self.nom_state

        # apply transformation
        u = np.clip(u, self.act_low, self.act_high)  #clip the action
        cur_zyx = cur_pose[-3:]
        cur_rot = R.from_euler('zyx', cur_zyx)  #get rotation from eular
        rot_mat_wc = cur_rot.as_matrix()  # R_wc

        translation = np.asarray([
            u[0] * math.sin(u[2]) * math.cos(u[1]),
            u[0] * math.sin(u[2]) * math.sin(u[1]), u[0] * math.cos(u[2])
        ])
        delta_rot = R.from_euler(
            'zyx', u[-3:])  # get applied translation and rotation from action

        new_xyz = cur_pose[0:3] + translation  #compute new translation

        rot_mat_cd = delta_rot.as_matrix()
        rot_mat_wd = np.dot(rot_mat_wc, rot_mat_cd)
        new_rot = R.from_matrix(rot_mat_wd)
        new_zyx = new_rot.as_euler('zyx')  #compute new rotation

        new_pose = np.concatenate([new_xyz, new_zyx])

        new_pose = np.clip(new_pose, self.state_low[-6:],
                           self.state_high[-6:])  #clip the state
        new_rot = R.from_euler('zyx', new_pose[-3:])  #get the rot again

        # apply projection model
        rot_mat_wd = new_rot.as_matrix()
        t_vec_wd = np.asarray(new_pose[0:3].reshape((-1, 1)))

        trans_mat_wd = np.concatenate([rot_mat_wd, t_vec_wd], axis=1)
        trans_mat_wd = np.concatenate(
            [trans_mat_wd, np.asarray([[0., 0., 0., 1]])],
            axis=0)  # get transformation mat T_wc = [r_wc,t_wc;0s,1]
        trans_mat_dw = np.linalg.inv(trans_mat_wd)  # take inverse to get T_cw

        square_d = np.dot(trans_mat_dw, self.square_points)

        h_img_uvs = np.dot(self.K, square_d[:3, :])
        img_uvs = np.divide(h_img_uvs[:2, :], h_img_uvs[2, :])
        self.img_square_points = img_uvs
    

        ## compute new coveragemin and coveragemax
        img_uv_center = np.mean(img_uvs, axis=1)

        n_center = np.divide(img_uv_center,
                             np.asarray([self.img_wid, self.img_hei
                                         ]))  # normalize the coordinate

        # get new set obs to record
        n_uvs = img_uvs / np.asarray([self.img_wid, self.img_hei
                                         ]).reshape((2, 1)) # 2 * 4
        new_obs = np.concatenate([n_uvs.reshape((-1,)), n_center, new_pose])
        #new_obs = np.concatenate([n_center, n_uvs, new_pose])
        self.set_state[:, self.count] = new_obs
        self.count += 1
        self.nom_state = new_pose

        #TODO: modify x, y coverage
        '''
        [U00min, V00min, U01min, V01max, U11max, V11max, U10max, V10min]
        '''

        new_U00min = self.state[0]
        new_V00min = self.state[1]
        new_U01min = self.state[2]
        new_V01max = self.state[3]
        new_U11max = self.state[4]
        new_V11max = self.state[5]
        new_U10max = self.state[6]
        new_V10min = self.state[7]

        if n_center[0] < 0.5 and n_center[1] < 0.5: # 00 part
            new_U00min = min(new_U00min, n_center[0])
            new_V00min = min(new_V00min, n_center[1])

        if n_center[0] < 0.5 and n_center[1] > 0.5: # 01 part
            new_U01min = min(new_U01min, n_center[0])
            new_V01max = max(new_V01max, n_center[1])

        if n_center[0] > 0.5 and n_center[1] > 0.5: # 11 part
            new_U11max = max(new_U11max, n_center[0])
            new_V11max = max(new_V11max, n_center[1])

        if n_center[0] > 0.5 and n_center[1] < 0.5: # 10 part
            new_U10max = max(new_U10max, n_center[0])
            new_V10min = min(new_V10min, n_center[1])

        #TODO: modify coverage for IMU
        '''
        dx_max, dy_max, dz_max, dalpha_max, dbeta_max, dgamma_max
        '''
        new_pose_max = self.state[12:18]

        diff_pose = np.divide(np.abs(new_pose - cur_pose), self.state_high[-6:] - self.state_low[-6:])

        comp = np.concatenate([new_pose_max.reshape((-1,1)), 
                                diff_pose.reshape((-1,1))],
                                axis=1)

        new_pose_max = np.max(comp, axis = 1)

        
        # area
        a = (img_uvs[:, 0] - img_uvs[:, 1]).reshape((-1, 1))
        b = (img_uvs[:, 2] - img_uvs[:, 1]).reshape((-1, 1))
        area = np.linalg.det(np.concatenate(
            [a, b], axis=1))  # compute area of the square in the image
        size = area / (self.img_wid * self.img_hei
                       )  # compute portion of square in the image
        new_size_min = min(self.state[8], size)
        new_size_max = max(self.state[9], size)  #compute size coverage

        # skew
        angle = math.acos(
            np.dot(a.T, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        skew = (math.pi - angle) / (math.pi)
        new_skew_min = min(self.state[10], skew)
        new_skew_max = max(self.state[11], skew)

        ## compute reward
        reward = self.state[0] - new_U00min + self.state[1] - new_V00min
        reward += self.state[2] - new_U01min + new_V01max - self.state[3]
        reward += new_U11max - self.state[4] + new_V11max - self.state[5]
        reward += new_U10max - self.state[6] + self.state[7] - new_V10min
        reward += new_size_max - self.state[9] + self.state[8] - new_size_min
        reward += new_skew_max - self.state[11] + self.state[10] - new_skew_min  # add coverage

        #TODO: add pose coverage
        reward += reward * 0.5 + np.sum(new_pose_max - self.state[12:18])

        
        reward -= abs(u[0]) * 0.1
        reward -= np.linalg.norm(u[-3:]) * 0.02  # minus the length

        reward *= 100

        # compute done
        done = new_U00min < 0.45 and new_V00min < 0.45
        done = done and new_U01min < 0.45 and new_V01max > 0.55
        done = done and new_U11max > 0.55 and new_V11max > 0.55
        done = done and new_U10max > 0.55 and new_V10min < 0.45
        done = done and (new_skew_max - new_skew_min) > 0.05 and (
            new_size_max - new_size_min) > 0.05
        done = done and (new_pose_max > 0.3).all()

        if done:
            reward += 200

        self.state = np.asarray([
            new_U00min,
            new_V00min,
            new_U01min,
            new_V01max,
            new_U11max,
            new_V11max,
            new_U10max,
            new_V10min,
            new_size_min,
            new_size_max,
            new_skew_min,
            new_skew_max,
        ])

        #TODO: add pose coverage
        self.state = np.concatenate([self.state, new_pose_max.reshape((-1,))])
        self.state = np.concatenate([self.state, new_pose])

        return self.set_state, self.nom_state, reward, done, {
            'set_state: ': self.set_state,
            'state: ': self.state,
            'nom_state: ': self.nom_state,
            'action: ': u,
            'img_uv_center': img_uv_center
        }

        
    def reset(self):
        '''
        reset: return to the origin, clear the coverage
        state = [0.5,0.5,0.5,0.5,size,size,0.5,0.5]
        compute initial area
        '''
        # set states
        self.set_state = []
        self.nom_state = np.zeros((6,))

        self.state = np.concatenate([np.ones((12, )) * 0.5, np.zeros((12, ))])

        h_img_uvs = np.dot(self.K, self.square_points[:3, :])
        img_uvs = np.divide(h_img_uvs[:2, :], h_img_uvs[2, :])

        a = (img_uvs[:, 0] - img_uvs[:, 1]).reshape((-1, 1))
        b = (img_uvs[:, 2] - img_uvs[:, 1]).reshape((-1, 1))
        area = np.linalg.det(np.concatenate(
            [a, b], axis=1))  # compute area of the square in the image
        size = area / (self.img_wid * self.img_hei)

        self.state[8] = size
        self.state[9] = size

        #initialize visualization
        self.state_hist = []
        self.img_square_points = self.img_uvs_init

        # init set state
        n_uvs = self.img_uvs_init / np.asarray([self.img_wid, self.img_hei
                                         ]).reshape((2, 1))
                                         
        obs = np.concatenate([n_uvs.reshape((-1,)), 0.5*np.ones((2,)), self.nom_state])
        #obs = np.concatenate([0.5*np.ones((2,)), n_uvs, self.nom_state])
        self.set_state = np.repeat(obs.reshape((-1, 1)), 11, axis=1)
        self.count = 1

        return self.set_state, self.nom_state

    def visualize(self):
        '''
        visualize the current state:
        '3D visualize of the camera pose and the square
        '2D image visualize
        '''
        if len(self.state_hist) == 0:
            # plt.show(block=True)
            plt.clf()
            self.ax = self.fig.add_subplot(2, 1, 1, projection='3d')
            self.ax2 = self.fig.add_subplot(2, 1, 2)
            

        #visualize the square

        square = np.concatenate(
            [self.square_points, self.square_points[:, 0].reshape((-1, 1))],
            axis=1)
        # print(square)
        self.ax.plot3D(square[0, :], square[1, :], square[2, :], 'gray')

        #visualize the camera
        cur_pose = self.state[-6:]

        self.ax.scatter3D(cur_pose[0],
                          cur_pose[1],
                          cur_pose[2],
                          'o',
                          color='green')  # plot position

        c_origin = cur_pose[:3]
        cur_zyx = cur_pose[-3:]
        cur_rot = R.from_euler('zyx', cur_zyx)  #get rotation from eular
        rot_mat_wc = cur_rot.as_matrix()  # R_wc
        z_one = np.dot(rot_mat_wc, np.asarray([0., 0., 0.2])) + c_origin

        direction = np.concatenate(
            [c_origin.reshape(
                (3, 1)), z_one.reshape((3, 1))], axis=1)

        self.ax.plot3D(direction[0, :], direction[1, :], direction[2, :],
                       'red')  #plot direction

        # visualize the trajectory
        self.state_hist.append(self.state.reshape((-1, 1)))
        hist_array = np.concatenate(self.state_hist, axis=1)
        self.ax.plot3D(hist_array[-6, :], hist_array[-5, :], hist_array[-4, :],
                       'black')

        self.ax.set_ylim((-0.5, 0.5))
        self.ax.set_xlim((-0.5, 0.5))
        self.ax.set_zlim((-0.5, 2))
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        #visualize the 2d points
        x = [[self.img_square_points[0, 0], self.img_square_points[0, 1]],
             [self.img_square_points[0, 1], self.img_square_points[0, 2]],
             [self.img_square_points[0, 2], self.img_square_points[0, 3]],
             [self.img_square_points[0, 3], self.img_square_points[0, 0]]]
        y = [[self.img_square_points[1, 0], self.img_square_points[1, 1]],
             [self.img_square_points[1, 1], self.img_square_points[1, 2]],
             [self.img_square_points[1, 2], self.img_square_points[1, 3]],
             [self.img_square_points[1, 3], self.img_square_points[1, 0]]]

        for i in range(len(x)):
            self.ax2.plot(x[i], y[i], color='r')
            self.ax2.scatter(x[i], y[i], color='b')

        self.ax2.xaxis.tick_top()
        self.ax2.yaxis.tick_left()
        self.ax2.set_xlim((0, 640))
        self.ax2.set_ylim((480, 0))

        plt.pause(0.5)