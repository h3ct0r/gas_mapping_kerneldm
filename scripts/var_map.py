#!/usr/bin/env python

import rospy
from PIL import Image

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import threading

class var_map():
    def __init__(self):
        self.pub = rospy.Publisher('VAR_array', MarkerArray, queue_size=1000)
        self.aux = 0
        self.aux2 = 0
        self.first = True
        self.update_rate = 5
        self.odom_x_positions = []
        self.odom_y_positions = []
        self.pub_rate = rospy.Time.from_sec(time.time()).to_sec()
        rospy.init_node('marker_pub', anonymous=True)
        self.rate = rospy.Rate(10) # 10hz
        self.marker_array_lock = threading.Lock()

        self.main()

    def map_subscriber(self,msg):
        self.origin = msg.info.origin.position
        self.data = msg.data
        self.robot_position = np.zeros((msg.info.width/10,msg.info.height/10))
        # self.blur = np.zeros((msg.info.width,msg.info.height))
        self.data_r = np.array(self.data).reshape((msg.info.height,msg.info.width)).astype('float32')
        self.resized = cv2.resize(self.data_r, (msg.info.width/10,msg.info.height/10), interpolation = cv2.INTER_AREA)

        self.update_map()

    def odom_subscriber(self,msg):
######## Nasa_v4
        #if self.robot_position[int(msg.pose.pose.position.x*5)][int(msg.pose.pose.position.y*5)] != 1:
        #    self.var_map(int(msg.pose.pose.position.x*5),int(msg.pose.pose.position.y*5))
######## MARPIRlab
        # t = rospy.Time.from_sec(time.time())
        # seconds = t.to_sec()
        if self.robot_position[int(msg.pose.pose.position.x*5)+27][int(msg.pose.pose.position.y*5)+30] != 1:
            self.var_map(int(msg.pose.pose.position.x*5)+27,int(msg.pose.pose.position.y*5)+30)
            if self.aux >= self.update_rate or self.first == True:
                self.update_map()
                self.aux=0

            self.aux +=1


    def var_map(self,x,y):

        if self.aux2 >=2 or self.first == True:
            for i in xrange(-5,6):
                for j in xrange(-5,6):
                    if (self.robot_position[x+i][y+j] < (1 -  (np.sqrt((i**2)+(j**2))/7.07))):
                        self.robot_position[x+i][y+j] = (1 -  (np.sqrt((i**2)+(j**2))/7.07))
            self.aux2 = 0
        self.aux2 += 1
        self.first = False



    def update_map(self):
        self.i=-1

        self.markerArray = MarkerArray()
        with self.marker_array_lock:
            
            for x in xrange(self.resized.shape[1]):
                    for y in xrange(self.resized.shape[0]):
                        self.i+=1
                        if self.resized[y][x] == 0:
                            self.marker = Marker()
                            # self.markerArray.markers = []
                            
                            self.marker.id = (self.i * self.resized.shape[0]) + y
                            self.marker.header.frame_id = "map"
                            self.point_msg = Point()

                            self.marker.type = self.marker.CUBE
                            self.marker.action = self.marker.ADD

                            self.marker.pose.position.x = (x * 0.2) + self.origin.x+0.1
                            self.marker.pose.position.y = (y * 0.2) + self.origin.y+0.1
                            self.marker.pose.orientation.w = 1
                            self.marker.lifetime = rospy.Duration()
                            self.marker.scale.x = 0.2
                            self.marker.scale.y = 0.2
                            self.marker.scale.z = 0.1
                            self.marker.color.a = 0.5
                            self.marker.color.b = 1.0 - self.robot_position[x][y]
                            self.marker.color.r = self.robot_position[x][y]
                            self.markerArray.markers.append(self.marker)

        self.pub.publish(self.markerArray)


    

    def main(self):

        self.markerArray = MarkerArray()

        while not rospy.is_shutdown():
            rospy.Subscriber("/map", OccupancyGrid, self.map_subscriber)
            rospy.Subscriber("/odom", Odometry, self.odom_subscriber)
            rospy.spin()

        
if __name__ == '__main__':
    try:
        a = var_map()
    except rospy.ROSInterruptException:
        pass
