#!/usr/bin/env python

import rospy
from threading import Lock
import numpy as np
import cv2
import time
import tf
import matplotlib.pyplot as plt
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from visualize_co2_markers import Co2Visualizer
from td_kernel_dmvw import TDKernelDMVW


class GasEstimationNode(object):
    def __init__(self):
        """
        Produce a rviz visualization of a gas distribution map from
        co2 measurements + locations + timestamps
        """

        self._gas_var_pub_topic_name = '/gas_var_markers'
        self._gas_mean_pub_topic_name = '/gas_mean_markers'

        self._min_sensor_val = 0.0
        self._max_sensor_val = 3.0
        self._frame_id = "/world"

        self.lock = Lock()
        self.listener = tf.TransformListener()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # publishers
        self.var_map_pub = rospy.Publisher(self._gas_var_pub_topic_name, MarkerArray, latch=True)
        self.mean_map_pub = rospy.Publisher(self._gas_mean_pub_topic_name, MarkerArray, latch=True)

        self.cell_size = 0.2
        self.kernel_size = 6 * self.cell_size
        self.wind_scale = 0.05
        self.time_scale = 0.001
        self.evaluation_radius = 6 * self.kernel_size

    def run(self):
        """ main entry point """

        def scale(im, nR, nC):
            """
            Scale image
            :param im: matrix image
            :param nR: n rows
            :param nC: n columns
            :return:
            """
            nR0 = len(im)  # source number of rows
            nC0 = len(im[0])  # source number of columns
            return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                     for c in range(nC)] for r in range(nR)]

        rate = rospy.Rate(0.5)

        co2viz = Co2Visualizer()

        while not rospy.is_shutdown():

            co2_data = co2viz.get_last_co2_pose_data()
            if not co2_data:
                print "no co2viz data"
                rate.sleep()
                continue

            if len(co2_data) < 3:
                print "measurements less than 3"
                continue

            # load data
            positions_x = []
            positions_y = []
            concentrations = []
            wind_directions = []
            wind_speeds = []
            timestamps = []

            for e in co2_data:
                pos = e['pos']
                positions_x.append(pos[0])
                positions_y.append(pos[1])
                concentrations.append(e['co2'])
                wind_directions.append(0)  # no wind
                wind_speeds.append(0)  # no wind

                timestamps.append(e['stamp_sec'])

            min_x = min(positions_x)
            min_y = min(positions_y)
            max_x = max(positions_x)
            max_y = max(positions_y)

            # call Kernel
            start_t = time.time()
            kernel = TDKernelDMVW(min_x, min_y, max_x, max_y, self.cell_size, self.kernel_size, self.wind_scale,
                                  self.time_scale, low_confidence_calculation_zero=True,
                                  evaluation_radius=self.evaluation_radius)

            kernel.set_measurements(positions_x, positions_y, concentrations, timestamps, wind_speeds, wind_directions)
            kernel.calculate_maps()
            end_t = time.time()
            print("elapsed kernel: {:.2f} secs, measurements:{}".format(end_t - start_t, len(positions_x)))

            local_cell_grid_x, local_cell_grid_y = np.mgrid[kernel.min_x:kernel.max_x:1, kernel.min_y:kernel.max_y:1]

            local_mean_map = np.array(scale(kernel.mean_map, int(kernel.max_x - kernel.min_x) + 1,
                                            int(kernel.max_y - kernel.min_y) + 1))
            where_are_NaNs = np.isnan(local_mean_map)
            local_mean_map[where_are_NaNs] = 0.0

            self.pub_gas_markers(self.mean_map_pub, local_mean_map, local_cell_grid_x, local_cell_grid_y)

            local_var_map = np.array(scale(kernel.variance_map, int(kernel.max_x - kernel.min_x) + 1,
                                            int(kernel.max_y - kernel.min_y) + 1))
            where_are_NaNs = np.isnan(local_var_map)
            local_var_map[where_are_NaNs] = 0.0

            self.pub_gas_markers(self.var_map_pub, local_var_map, local_cell_grid_x, local_cell_grid_y)

            # # Show result as map
            # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            # fig.suptitle('Kernel DM+V')
            #
            # ax1.set_aspect(1.0)
            # ax1.title.set_text("mean map")
            # ax1.contourf(kernel.cell_grid_x, kernel.cell_grid_y, kernel.mean_map)
            #
            # ax2.set_aspect(1.0)
            # ax2.title.set_text("variance map")
            # ax2.contourf(kernel.cell_grid_x, kernel.cell_grid_y, kernel.variance_map)
            #
            # ax3.set_aspect(1.0)
            # ax3.title.set_text("confidence map")
            # ax3.contourf(kernel.cell_grid_x, kernel.cell_grid_y, kernel.confidence_map)
            #
            # #plt.show()
            # plt.savefig('/tmp/kernel_dmv_local.png')
            # #plt.close()

            rate.sleep()

    def pub_gas_markers(self, map_publisher, local_map, local_cell_grid_x, local_cell_grid_y):

        markerArray = MarkerArray()

        max_v = np.max(local_map)

        width, height = local_cell_grid_x.shape

        c_id = 0
        for i in xrange(width):
            for j in xrange(height):

                marker = Marker()
                marker.header.frame_id = self._frame_id
                marker.id = c_id
                marker.type = marker.CUBE
                marker.action = marker.ADD
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 0.05
                marker.color.a = 0.7

                if local_map[i][j] == 0.0:  # prevent NaN
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                    continue
                else:
                    marker.color.r = 1.0 * local_map[i][j] / max_v
                    marker.color.g = 0.8 * (1 - 2.0 * abs(local_map[i][j] - max_v / 2) / max_v)
                    marker.color.b = 1 - 1.0 * local_map[i][j] / max_v

                marker.pose.orientation.w = 1.0

                wx = local_cell_grid_x[i][j]
                wy = local_cell_grid_y[i][j]

                marker.pose.position.x = wx
                marker.pose.position.y = wy
                marker.pose.position.z = 0.1
                markerArray.markers.append(marker)
                c_id += 1

        map_publisher.publish(markerArray)


def main():
    rospy.init_node('gas_estimation_node')
    gas_node = GasEstimationNode()
    gas_node.run()


if __name__ == '__main__':
    main()
