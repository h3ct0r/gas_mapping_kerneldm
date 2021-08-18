#!/usr/bin/env python

import rospy
from threading import Lock
import tf
import json
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class PublishGTNode(object):
    def __init__(self):
        self._gt_pub_topic_name = '/gt_markers'
        self._frame_id = rospy.get_param('~frame_id', '/world')
        self.gt_file_path = rospy.get_param('~file_path')

        self.lock = Lock()
        self.listener = tf.TransformListener()

        # publishers
        self.gt_pub = rospy.Publisher(self._gt_pub_topic_name, MarkerArray, latch=True)

        # prepare data
        self.gt_data = []
        with open(self.gt_file_path) as json_file:
            data = json.load(json_file)
            for e in data['artifacts']:
                if e['type'] == 'gas':
                    self.gt_data.append(e)

    def run(self):
        """ main entry point """

        rate = rospy.Rate(0.5)

        while not rospy.is_shutdown():
            self.pub_gt_markers()
            rate.sleep()

    def pub_gt_markers(self):

        markerArray = MarkerArray()

        c_id = 0
        for e in self.gt_data:

            marker = Marker()
            marker.header.frame_id = self._frame_id
            marker.id = c_id
            marker.type = marker.CUBE
            marker.action = marker.ADD
            marker.scale.x = 3.0
            marker.scale.y = 3.0
            marker.scale.z = 0.05
            marker.color.a = 0.7

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0

            marker.pose.orientation.w = 1.0

            marker.pose.position.x = e['x']
            marker.pose.position.y = e['y']
            marker.pose.position.z = e['z']
            markerArray.markers.append(marker)
            c_id += 1

            marker = Marker()
            marker.header.frame_id = self._frame_id
            marker.id = c_id
            marker.type = marker.TEXT_VIEW_FACING
            marker.text = "GAS {}".format(e['id'])
            marker.action = marker.ADD
            marker.scale.x = 2.0
            marker.scale.y = 2.0
            marker.scale.z = 2.0
            marker.color.a = 1.0

            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0

            marker.pose.orientation.w = 1.0

            marker.pose.position.x = e['x']
            marker.pose.position.y = e['y']
            marker.pose.position.z = e['z'] + 0.5
            markerArray.markers.append(marker)
            c_id += 1

        self.gt_pub.publish(markerArray)


def main():
    rospy.init_node('gt_estimation_node')
    gt_node = PublishGTNode()
    gt_node.run()


if __name__ == '__main__':
    main()
