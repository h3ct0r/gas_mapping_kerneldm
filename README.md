# ROS Gas Distribution Mapping with Kernel DM+V

Gas mapping system integrated with ROS using the KernelDM+V method (python2.7 compatible, tested with ROS Kinetic under Ubuntu 16.04)

### Original implementation 

Stephan Muller - https://github.com/smueller18/TDKernelDMVW

### Implementation based on the following papers:

- S. Asadi and A. Lilienthal, "Approaches to time-dependent gas distribution modelling," 2015 European Conference on Mobile Robots (ECMR), Lincoln, 2015, pp. 1-6.
- A. J. Lilienthal, M. Reggente, M. Trincavelli, J. L. Blanco and J. Gonzalez, "A statistical approach to gas distribution modelling with mobile robots - The Kernel DM+V algorithm," 2009 IEEE/RSJ International Conference on Intelligent Robots and Systems, St. Louis, MO, 2009, pp. 570-576.
- Neumann, Patrick. (2013). BAM-Dissertationsreihe. Bd. 109: Gas Source Localization and Gas Distribution Mapping with a Micro-Drone. Berlin : Bundesanstalt für
Materialforschung und -prüfung (BAM)

### Demo with GADEN and gas simulations:

Simulation environment: https://github.com/MAPIRlab/gaden

![gas_mapping_example](https://user-images.githubusercontent.com/2656938/129972728-6c634332-b1ba-4b58-b54c-9e443aa52d6c.gif)

### Depends on:

- core_messages/hw_msgs
- Python2 dependencies:
    - matplotlib
    - numpy
    - networkx

### How to run:

Load the rosbags:

`rosbag play spot2_globalplan_2020-02-26-12-51-20_0.bag spot2_artifact_2020-02-26-12-51-19_0.bag spot2_localplan_2020-02-26-12-51-20_0.bag spot2_artifact_2020-02-26-13-36-30_1.bag   spot2_tf_2020-02-26-12-51-20_0.bag base1_lamp_2020-02-26-13-55-29_102.bag lamp_posegraph.bag lamp_posegraph_incremental.bag -r 20 -s 1450`

Load the environment:

`roslaunch gas_mapping_kerneldm gas_map_bags_urban.launch`

### main_density_map_node.py

This node reads the `/spot2/co2` data, align the position of the measurements with the pose graph and then publishes a mean and variance map of the readings using the KernelDM+V method.

Subscribes to: 

- /spot2/co2
    - hw_msgs/PointSourceDetection
    
- /base1/lamp/pose_graph
    - pose_graph_msgs/PoseGraph

Publishes:

- /spot2/co2_marker
    - visualization_msgs/MarkerArray
    
- /gas_var_markers
    - visualization_msgs/MarkerArray
    
- /gas_mean_markers
    - visualization_msgs/MarkerArray