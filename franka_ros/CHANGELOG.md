# CHANGELOG

## 0.7.0 - UNRELEASED

Requires `libfranka` >= 0.5.0

  * Moved `panda_moveit_config` to [`ros-planning`](https://github.com/ros-planning/panda_moveit_config)
  * Publish `robot_mode` in `franka_state_controller`

## 0.6.0 - 2018-08-08

Requires `libfranka` >= 0.5.0

  * **BREAKING** Fixes for MoveIt, improving robot performance:
    * Fixed joint velocity and acceleration limits in `joint_limits.yaml`
    * Use desired joint state for move group
  * **BREAKING** Updated joint limits in URDF
  * **BREAKING** Fixed velocity, acceleration and jerk limits in `franka_hw`
  * **BREAKING** Start `franka_gripper_node` when giving `load_gripper:=true` to `franka_control.launch`
  * Allow to configure rate limiting, filtering and internal controller in `franka_control_node`
  * **BREAKING** `FrankaHW::FrankaHW` takes additional parameters.
  * **BREAKING** Enabled rate limiting and low-pass filtering by default (`franka_control_node.yaml`)
  * Publish desired joint state in `/joint_state_desired`
  * Removed `effort_joint_trajectory_controller` from `default_controllers.yaml`
  * Fixed a bug when switching between controllers using the same `libfranka` interface

## 0.5.0 - 2018-06-28

Requires `libfranka` >= 0.4.0

  * **BREAKING** Updated URDF:
    * Adjusted maximum joint velocity
    * Updated axis 4 hard and soft limits

## 0.4.1 - 2018-06-21

Requires `libfranka` >= 0.3.0

  * Added some missing includes to `franka_hw`
  * Add support for commanding elbow in Cartesian pose and Cartesian velocity interfaces

## 0.4.0 - 2018-03-26

Requires `libfranka` >= 0.3.0

  * **BREAKING** Removed `arm_id` and default `robot_ip` from launchfiles
  * **BREAKING** Changed namespace of `franka_control` controller manager
  * **BREAKING** Changed behavior of `gripper_action` for compatibility with MoveIt
  * Changes in `panda_moveit_config`:
    * Updated joint limits from URDF
    * Removed `home` poses
    * Fixed fake execution
    * Add `load_gripper` argument (default: `true`) to `panda_moveit.launch`
    * Conditionally load controllers/SRDFs based on `load_gripper`
    * Add gripper controller configuration (requires running `franka_gripper_node`)
  * Added `mimic` tag for gripper fingers to URDF and fixed velocity limits

## 0.3.0 - 2018-02-22

Requires `libfranka` >= 0.3.0

  * **BREAKING** Changed signatures in `franka_hw::FrankaModelHandle`
  * **BREAKING** Added epsilon parameters to `franka_gripper/Grasp` action
  * Added Collada meshes for Panda and Hand
  * Added missing dependencies to `panda_moveit_config` and `franka_example_controllers`
  * Fixed linker errors when building with `-DFranka_DIR` while an older version of
    `ros-kinetic-libfranka` is installed
  * Added gripper joint state publisher to `franka_visualization`
  * Moved `move_to_start.py` example script to `franka_example_controllers`

## 0.2.2 - 2018-01-31

Requires `libfranka` >= 0.2.0

  * Catkin-related fixes for `franka_example_controllers`
  * Added missing `<build_export_depend>` for `message_runtime`

## 0.2.1 - 2018-01-30

Requires `libfranka` >= 0.2.0

  * Added missing dependency to `franka_example_controllers`
  * Lowered rotational gains for Cartesian impedance example controller

## 0.2.0 - 2018-01-29

Requires `libfranka` >= 0.2.0

  * Added missing run-time dependencies to `franka_description` and `franka_control`
  * Added `tau_J_d`, `m_ee`, `F_x_Cee`, `I_ee`, `m_total`, `F_x_Ctotal`, `I_total`,
    `theta` and `dtheta` to `franka_msgs/FrankaState`
  * Added new errors to `franka_msgs/Errors`
  * Updated and improved examples in `franka_example_controllers`
  * Fixed includes for Eigen3 in `franka_example_controllers`
  * Fixed gripper state publishing in `franka_gripper_node`

## 0.1.2 - 2017-10-10

  * Fixed out-of-workspace build

## 0.1.1 - 2017-10-09

  * Integrated `franka_description` as subdirectory
  * Fixed dependencies on libfranka
  * Fixed RViz config file paths
  * Added missing `test_depend` to `franka_hw`
  * Added missing CMake install rules

## 0.1.0 - 2017-09-15

  * Initial release

