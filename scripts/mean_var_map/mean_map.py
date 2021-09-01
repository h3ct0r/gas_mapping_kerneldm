#!/usr/bin/env python

import rospy
from PIL import Image

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
from olfaction_msgs.msg import gas_sensor
import matplotlib.pyplot as plt
from ros_waspmote_reader.msg import wasp
import tf
from td_kernel_dmvw import TDKernelDMVW
import cv2
import time
import numpy as np
import threading


class Gas_Estimation_Map():
    def __init__(self):
        self.mean_publisher = rospy.Publisher('mean_array', MarkerArray, queue_size=1000)
        self.confidence_publisher = rospy.Publisher('confidence_array', MarkerArray, queue_size=1000)
        self._frame_id = "/map"
        self.aux = 0
        self.first = True
        self.cell_size = 0.2
        self.kernel_size = 4 * self.cell_size
        self.wind_scale = 0.05
        self.time_scale = 0.001
        self.evaluation_radius = 4 * self.kernel_size

        self.x_reads = []
        self.y_reads = []
        self.concentration = []
        self.wind_directions = []
        self.wind_speeds = []
        self.timestamps = []
        self.time = time.time()
        self.pub_rate = rospy.Time.from_sec(time.time()).to_sec()
        rospy.init_node('mean_pub', anonymous=True)
        self.listener = tf.TransformListener() 
        self.marker_array_lock = threading.Lock()

        self.main()

    def map_subscriber(self,msg):

        self.origin = msg.info.origin.position
        self.data = msg.data

        if self.first == True:
            self.kernel = TDKernelDMVW(self.origin.x, self.origin.y,
                                       self.origin.x+msg.info.width*msg.info.resolution, self.origin.y+msg.info.height*msg.info.resolution,
                                       self.cell_size, self.kernel_size,
                                       self.wind_scale, self.time_scale,
                                       low_confidence_calculation_zero=True, evaluation_radius = self.evaluation_radius)
            self.data_r = np.array(self.data).reshape((msg.info.height,msg.info.width)).astype('float32')
            self.resized = cv2.resize(self.data_r, (self.kernel.number_of_x_cells,self.kernel.number_of_y_cells), interpolation = cv2.INTER_AREA)
            self.update_map('empty_map')
            self.first = False
        else:
            self.data_r = np.array(self.data).reshape((msg.info.height,msg.info.width)).astype('float32')
            self.resized = cv2.resize(self.data_r, (self.kernel.number_of_x_cells,self.kernel.number_of_y_cells), interpolation = cv2.INTER_AREA)
            self.kernel = TDKernelDMVW(self.origin.x, self.origin.y,
                                       self.origin.x+msg.info.width*msg.info.resolution, self.origin.y+msg.info.height*msg.info.resolution,
                                       self.cell_size, self.kernel_size,
                                       self.wind_scale, self.time_scale,
                                       low_confidence_calculation_zero=True, evaluation_radius = self.evaluation_radius)

            self.kernel.set_measurements(self.x_reads,self.y_reads,self.concentration,self.timestamps,self.wind_speeds,self.wind_directions)
            self.kernel.calculate_maps()
            self.update_map('mean_array')
            self.update_map('confidence_array')

    def handle_gas_sensor_cb(self, msg):
        self.aux+= 1
        if self.aux >=5: # The sensor message rate its 5 hz
            try:
                (trans, rot) = self.listener.lookupTransform(self._frame_id, msg.header.frame_id, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("[GMRF] cannot find transformation between gas sensor and map frame_id")
                return

            self.x_reads.append(((trans[0]))) #+ (self.origin.x)))
            self.y_reads.append(((trans[1]))) #+ (self.origin.y)))

            self.concentration.append(msg.raw)

            self.wind_directions.append(0)
            self.wind_speeds.append(0)
            self.timestamps.append(time.time() - self.time)

            if (rospy.Time.from_sec(time.time()).to_sec() - self.pub_rate) >=1 :
                start_t = time.time()        
                self.kernel.set_measurements(self.x_reads,self.y_reads,self.concentration/np.max(self.concentration),self.timestamps,self.wind_speeds,self.wind_directions)
                self.kernel.calculate_maps()
                end_t = time.time()
                # print("elapsed kernel: {:.2f} secs, measurements:{}".format(end_t - start_t, len(self.x_reads)))

                mean_where_are_NaNs = np.isnan(self.kernel.mean_map)
                self.kernel.mean_map[mean_where_are_NaNs] = 0.0
                confidence_where_are_NaNs = np.isnan(self.kernel.confidence_map)
                self.kernel.confidence_map[confidence_where_are_NaNs] = 0.0
                self.update_map('mean_array')
                self.update_map('confidence_array')
                self.pub_rate = rospy.Time.from_sec(time.time()).to_sec()

            self.aux = 0


    def update_map(self,pub_name):

        self.i=-1
        max_mean = np.max(self.kernel.mean_map)
        max_confidence = np.max(self.kernel.confidence_map)

        with self.marker_array_lock:
            self.markerArray = MarkerArray()

            for x in xrange(0,self.resized.shape[1]):
                for y in xrange(0,self.resized.shape[0]):

                        self.i+=1
                        if self.resized[y,x] == 0:
                            self.marker = Marker()
                            
                            self.marker.id = (self.i * self.resized.shape[0]) + y
                            self.marker.header.frame_id = self._frame_id

                            self.marker.type = self.marker.CUBE
                            self.marker.action = self.marker.MODIFY

                            self.marker.pose.position.x = (self.origin.x*self.cell_size) + (self.kernel.cell_grid_x[x,y] ) + 1.2
                            self.marker.pose.position.y = (self.origin.y*self.cell_size) + (self.kernel.cell_grid_y[x,y] )+ 1.2
                            self.marker.pose.orientation.w = 1
                            self.marker.lifetime = rospy.Duration()
                            self.marker.scale.x = self.cell_size
                            self.marker.scale.y = self.cell_size
                            self.marker.scale.z = 0.1
                            self.marker.color.a = 0.5
                            
                            if (pub_name == 'mean_array'):
                                self.marker.color.r = (1.0 * self.kernel.mean_map[x][y]) / max_mean
                                self.marker.color.g = 0.8 * (1 - 2.0 * abs(self.kernel.mean_map[x][y] - max_mean / 2) / max_mean)
                                self.marker.color.b = 1 - (1.0 * self.kernel.mean_map[x][y] / max_mean)
                            elif (pub_name == 'confidence_array'):
                                self.marker.color.r = (1.0 * self.kernel.confidence_map[x][y]) / max_confidence
                                self.marker.color.g = 0.8 * (1 - 2.0 * abs(self.kernel.confidence_map[x][y] - max_confidence / 2) / max_confidence)
                                self.marker.color.b = 1 - (1.0 * self.kernel.confidence_map[x][y] / max_confidence)
                            elif (pub_name == 'empty_map'):
                                self.marker.color.r = 0.0
                                self.marker.color.g = 0.0
                                self.marker.color.b = 1.0
                                

                            self.markerArray.markers.append(self.marker)


        if (pub_name == 'mean_array'):
            self.mean_publisher.publish(self.markerArray)
        elif (pub_name == 'confidence_array'):
            self.confidence_publisher.publish(self.markerArray)
        elif (pub_name == 'empty_map'):
            self.confidence_publisher.publish(self.markerArray)
            self.mean_publisher.publish(self.markerArray)
        
    

    def main(self):

        self.markerArray = MarkerArray()

        while not rospy.is_shutdown():
            rospy.Subscriber("/map", OccupancyGrid, self.map_subscriber)
            # rospy.Subscriber("/espeleo_gas_pub", wasp, self.gas_cb) # For seekur test
            rospy.Subscriber('/PID/Sensor_reading',gas_sensor,self.handle_gas_sensor_cb)
            rospy.spin()

        
if __name__ == '__main__':
    try:
        map = Gas_Estimation_Map()
    except rospy.ROSInterruptException:
        pass
