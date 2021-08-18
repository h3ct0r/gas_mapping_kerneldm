#!/usr/bin/env python

import sys
import os
import rospy
from hw_msgs.msg import PointSourceDetection
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from pose_graph_msgs.msg import PoseGraph, PoseGraphNode
import matplotlib
from matplotlib import cm
import tf
import threading
from pose_graph_tools import nodeKeyToRobot


class Co2Visualizer:
    def __init__(self):
        """
        Publish the basic co2 markers and update the position of
        known co2 measurements with the pose graph
        """

        self.robot_namespace = '/spot2/'
        self.robot = self.robot_namespace.replace("/", "")
        self.signal_sub = rospy.Subscriber(
            self.robot_namespace + "co2",
            PointSourceDetection,
            self.co2_callback,
            queue_size=1000)

        self.pose_graph_sub = rospy.Subscriber("/base1/lamp/pose_graph",
            PoseGraph,
            self.pose_graph_callback,
            queue_size=10)

        self.pub = rospy.Publisher(
            self.robot_namespace +
            'co2_marker',
            MarkerArray,
            queue_size=10,
            latch=True)

        self.listener = tf.TransformListener()
        self.id = 0
        self.marker_array = MarkerArray()
        self.ordered_nodes = []
        self.marker_array_lock = threading.Lock()
        self.key_to_node = {}

        self.last_co2_pose_data = []
        
    def create_co2_marker(self, msg):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = msg.header.stamp
        marker.ns = self.robot + "_co2"
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.pose.orientation.w = 1
        marker.text = str(msg.strength)

        # Scale and color based on strength
        # Small and blue and low strength
        # Big and red is high strength
        max_strength = 2000
        min_strength = 200
        scale = np.clip(
            float(
                msg.strength -
                min_strength) /
            float(
                max_strength -
                min_strength),
            0.0,
            1.0)
        if scale == 0.0:
            return None

        scale_multiplier = 3.5
        marker.type = Marker.SPHERE
        marker.id = self.id
        marker.lifetime.secs = 0
        marker.scale.x = scale * scale_multiplier
        marker.scale.y = scale * scale_multiplier
        marker.scale.z = scale * scale_multiplier

        # Use winter color scheme (blue to green)
        # Should be distinct from co2 markers, which use winter
        cmap = matplotlib.cm.get_cmap('winter')
        color = cmap(scale)
        marker.color.a = 0.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        self.id += 1
        return marker

    def co2_callback(self, msg):
        # Add CO2 marker to marker_array_message
        # Position is not updated until pose graph callback
        #rospy.loginfo("CO2: " + str(msg.header.stamp.to_sec()) + " " + str(msg.strength))

        marker = self.create_co2_marker(msg)
        if marker is not None:
            with self.marker_array_lock:
                self.marker_array.markers.append(marker)

    def lookup_pose(self, stamp):
        if len(self.ordered_nodes) == 0:
            return None
        closest_node = self.ordered_nodes[0]
        smallest_delta = abs(stamp.to_sec() - closest_node.header.stamp.to_sec())
        closest_index = 0
        for i in range(0, len(self.ordered_nodes)):
            node = self.ordered_nodes[i]
            time_delta = abs(stamp.to_sec() - node.header.stamp.to_sec())
            if time_delta < smallest_delta:
                smallest_delta = time_delta
                closest_node = node
                closest_index = i
        if (smallest_delta > 5.0) and ((closest_index == 0) or (closest_index == len(self.ordered_nodes) -1)):
            return None
        if (closest_index == 0) or (closest_index == len(self.ordered_nodes) -1):
            return (closest_node.pose.position.x, closest_node.pose.position.y, closest_node.pose.position.z)

        next_node = self.ordered_nodes[closest_index + 1]
        prev_node = self.ordered_nodes[closest_index - 1]
        next_time = next_node.header.stamp.to_sec()
        prev_time = prev_node.header.stamp.to_sec()
        query_time = stamp.to_sec()
        closest_time = closest_node.header.stamp.to_sec()
        x = 0
        y = 0
        z = 0
        if query_time > closest_time: 
            frac = (query_time - closest_time) / (next_time - closest_time)
            x = (1-frac)*closest_node.pose.position.x + (frac)*next_node.pose.position.x
            y = (1-frac)*closest_node.pose.position.y + (frac)*next_node.pose.position.y
            z = (1-frac)*closest_node.pose.position.z + (frac)*next_node.pose.position.z
        elif query_time <= closest_time: 
            frac = (query_time - prev_time) / (closest_time - prev_time)
            x = (1-frac)*prev_node.pose.position.x + (frac)*closest_node.pose.position.x
            y = (1-frac)*prev_node.pose.position.y + (frac)*closest_node.pose.position.y
            z = (1-frac)*prev_node.pose.position.z + (frac)*closest_node.pose.position.z
        return (x,y,z)

    def pose_graph_callback(self, msg):
        for node in msg.nodes:
            if nodeKeyToRobot(node.key) == self.robot:
                # self.ordered_nodes.append(node)
                if node.key in self.key_to_node:
                    self.key_to_node[node.key].pose.position.x = node.pose.position.x
                    self.key_to_node[node.key].pose.position.y = node.pose.position.y
                    self.key_to_node[node.key].pose.position.z = node.pose.position.z
                else:
                    self.key_to_node[node.key] = node
        
        # Check for subscribers
        if self.pub.get_num_connections() == 0:
            rospy.loginfo("No subscribers: skipping marker updates")
            return
        #rospy.loginfo("Updating markers with new pose graph")

        # self.ordered_nodes.sort(key=lambda x : x.header.stamp.to_sec())
        self.ordered_nodes = sorted(self.key_to_node.values(), key=lambda x : x.header.stamp.to_sec())
        
        with self.marker_array_lock:
            local_co2_pose_data = []

            for i in range(0,len(self.marker_array.markers)):
                trans = self.lookup_pose(self.marker_array.markers[i].header.stamp)
                if trans is None:
                    self.marker_array.markers[i].color.a = 0.0
                else:
                    self.marker_array.markers[i].color.a = 0.5
                    self.marker_array.markers[i].pose.position.x = trans[0]
                    self.marker_array.markers[i].pose.position.y = trans[1]
                    self.marker_array.markers[i].pose.position.z = trans[2]

                    # update the location of the measurements
                    local_co2_pose_data.append(
                        {
                            'co2': float(self.marker_array.markers[i].text),
                            'pos': (trans[0], trans[1], trans[2]),
                            'stamp_sec': self.marker_array.markers[i].header.stamp.to_sec()
                        }
                    )

            self.last_co2_pose_data = local_co2_pose_data
            self.pub.publish(self.marker_array)

        # with open("/tmp/gas_data_spot2.txt", 'w') as f:
        #     for m in self.marker_array.markers:
        #         x, y = m.pose.position.x, m.pose.position.y
        #         co2_value = float(m.text)
        #         if x != 0.0 and y != 0.0:
        #             f.write('{} {} {}\n'.format(x, y, co2_value))

    def get_last_co2_pose_data(self):
        return self.last_co2_pose_data


def main(args):
    rospy.init_node('co2_visualizer', anonymous=True)
    viz = Co2Visualizer()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
