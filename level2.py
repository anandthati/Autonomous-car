#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import sensor_msgs.msg
import random
import numpy as np
from geometry_msgs.msg import Twist
from itertools import *
from operator import itemgetter
import os
from nav_msgs.msg import Odometry

import tf

def odom_callback(msg):
    # print("position:",msg.pose.pose.position)
    # print("orientation:",msg.pose.pose.orientation)
    eps = PI/180
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    ## yaw relects orientation of the bot
    # print("yaw:",yaw,"orientation:",msg.pose.pose.orientation)
    global orientation, curr_x, curr_y, deviation_running
    curr_y = msg.pose.pose.position.y
    curr_x = msg.pose.pose.position.x
    orientation = yaw
    ## the right turn may not be perfect and even a new orientation of 88 degrees will change the bot's course
    ## Hence, we implement a small program to correct its path
    


def LaserScanProcess(data):
    global angular_z,linear_x,orientation,PI
    LD = np.array(data.ranges)
    eps = PI/180
#    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
    np.save("LD.npy",LD)
    # get the value for linear and angular velocities
    #LD = msg
    S = np.argsort(LD).flatten()
    S = np.hstack((S[S<180],S[S >= 180]-360))
    S = S[np.argwhere(LD[S]>5).flatten()]
    a = np.min(S[S>0])
    b = np.max(S[S<=0])
    if abs(a)> abs(b):
        ang = b+2
    else :
        ang = a+2
    #print (LD,ang,orientation,angular_z)
    #angular_z = ang
    





    ##################################################################################
    angular_speed = 30*(PI/180)
    target = ang
    if target > 0 :
        angular_z = abs(angular_speed) #anticlock
    elif target < 0 :
        angular_z = -abs(angular_speed) #clockwise

    while( abs((orientation) - target) > eps):
        # print("orientation:", orientation,"target:",target)
        #vel_msg = vel_msg_init()
        if orientation<target:
            angular_z = abs(angular_speed)
            # vel_msg.angular.z = max(10*PI/180,2*abs(orientation-target))
        else:
            angular_z = -abs(angular_speed)
        #print(angular_z,abs((orientation) - target))
            # vel_msg.angular.z = min(-10*PI/180,-2*abs(orientation-target))
        #velocity_publisher.publish(vel_msg)
        linear_x = 0.4
    
    #angular_z = 10

def controller():
    global angular_z,linear_x
    rospy.init_node('listener', anonymous=True)

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber("scan", sensor_msgs.msg.LaserScan , LaserScanProcess)
    odom_sub = rospy.Subscriber(odometry_topic, Odometry, odom_callback)
    rate = rospy.Rate(10) # 10hz
    rospy.spin()
    while not rospy.is_shutdown():
        command = Twist()
        command.linear.x = linear_x
        command.angular.z = angular_z
        pub.publish(command)
        rate.sleep()

def main():
    os.system('gnome-terminal -x ' 'roslaunch hackathon level2.launch')    ##open the map with the bot and the world
    # time.sleep(5)
    try:
        controller()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    linear_x = 10
    odometry_topic = "/odom"
    angular_z = 10
    PI = 3.14
    orientation = PI/2
    main()
