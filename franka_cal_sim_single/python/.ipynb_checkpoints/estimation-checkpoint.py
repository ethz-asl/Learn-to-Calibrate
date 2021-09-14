#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic


import numpy as np
import tensorflow as tf

#define data structures
image_data = []
imu_data = []
state_data = []

#camera intrinsics
fx = tf.Variable(500.0)
fy = tf.Variable(500.0)
cx = tf.Variable(300.0)
cy = tf.Variable(200.0)
K = tf.sparse.SparseTensor(indices=[[0,0],[1,1],[2,2],[0,2],[1,2]],values=[fx,fy,1,cx,cy],dense_shape=[3,3])
#imu intrinsics
#camera_imu_extend
r = tf.Variable(0.0)
p = tf.Variable(0.0)
y = tf.Variable(0.0)
t = tf.Variable([[0.0],[0.0],[0.0]])
C_r = tf.sparse.SparseTensor(indices=[[0,0],[1,1],[1,2],[2,1],[2,2]],values=[1.0,tf.math.cos(r),-tf.math.sin(r),tf.math.sin(r),tf.math.cos(r)],dense_shape=[3,3])
C_p = tf.sparse.SparseTensor(indices=[[0,0],[0,2],[1,1],[2,0],[2,2]],values=[tf.math.cos(p),tf.math.sin(p),1.0,-tf.math.sin(r),tf.math.cos(r)],dense_shape=[3,3])
C_y = tf.sparse.SparseTensor(indices=[[0,0],[0,1],[1,0],[1,1],[2,2]],values=[tf.math.cos(y),-tf.math.sin(y),tf.math.sin(y),tf.math.cos(y),1.0],dense_shape=[3,3])
T_ci = tf.matmul(tf.matmul(tf.sparse_tensor_to_dense(C_y),tf.sparse_tensor_to_dense(C_p)),tf.sparse_tensor_to_dense(C_r))
T_ci = tf.concat([T_ci,t],1)
T_ci = tf.concat([T_ci,tf.constant([[0.0,0.0,0.0,1.0]])],0)
sess = tf.compat.v1.Session()
sess.run(tf.initialize_all_variables())
with tf.Session() as sess:
    print(sess.run(C_r))
    print(sess.run(C_p))
    print(sess.run(T_ci))
