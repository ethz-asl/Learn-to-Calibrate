# Franka Emika Panda MoveIt! Config Package

The Panda robot is the flagship MoveIt! integration robot used in the MoveIt! tutorials.
Any changes to MoveIt! need to be propagated into this config fast, so this package
is co-located under the ``ros-planning`` Github organization here.

## My Changes

In order to integrate the panda robot into gazebo I made changes to the following files:

- panda_moveit_config/config/panda_controllers.yaml
- panda_moveit_config/config/panda_gripper_controllers.yaml
- panda_moveit_config/launch/moveit.rviz