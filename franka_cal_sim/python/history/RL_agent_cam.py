#!/usr/bin/env python


import os
import rospy

python_path = rospy.get_param("/rl_client/python_path")
os.chdir(python_path)
for i in range(10):
    os.system("python3 RL_agent_sac.py")