# ROS integration for Franka Emika research robots

[![Build Status][travis-status]][travis]

See the [Franka Control Interface (FCI) documentation][fci-docs] for more information.

## My Changes

In order to integrate the panda robot into gazebo I made changes to the following files:

- franka_description/robots/hand.xacro
- franka_description/robots/panda_arm.xacro
- franka_description/robots/panda_arm_hand.urdf.xacro

I added the following files to the repository:

- franka_description/robots/panda.gazebo.xacro
- franka_description/robots/panda.transmission.xacro

## License

All packages of `franka_ros` are licensed under the [Apache 2.0 license][apache-2.0].

[apache-2.0]: https://www.apache.org/licenses/LICENSE-2.0.html
[fci-docs]: https://frankaemika.github.io/docs
[travis-status]: https://travis-ci.org/frankaemika/franka_ros.svg?branch=kinetic-devel
[travis]: https://travis-ci.org/frankaemika/franka_ros
