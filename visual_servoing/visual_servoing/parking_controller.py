#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.time import Time
from rcl_interfaces.msg import SetParametersResult
from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic")
        DRIVE_TOPIC = self.get_parameter("drive_topic").value # set in launch file; different for simulator vs racecar

        #  Boundary for when the car is considered pointed in the right direction
        self.declare_parameter("anglular_error_threshold", np.pi/6)
        self.angular_error_threshold = self.get_parameter("anglular_error_threshold").value
        
         # Boundaries for when the car is considered too close/too far from the cone for 
         # angle adjustment manuevers (negative is too close)
        self.declare_parameter("distance_error_thresholds", [-0.2, 0.2])
        self.distance_error_threshold = self.get_parameter("distance_error_thresholds").get_parameter_value().double_array_value

        # How fast we want the car to go while parking
        self.declare_parameter("parking_velocity", 0.8)
        self.parking_velocity = self.get_parameter("parking_velocity").value


        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.create_subscription(ConeLocation, "/relative_cone", 
            self.relative_cone_callback, 1)

        self.declare_parameter("parking_distance", 0.75)
        self.parking_distance = self.get_parameter("parking_distance").value

        self.relative_x = 0
        self.relative_y = 0
        self.reverse = False
        self.previous_x_error = 0 
        self.previous_time = Time()
        self.integrated_dist_error = 0

        self.integral_max, self.integral_min = -10, 10

        self.get_logger().info("Parking Controller Initialized")

        # Enable setting parking_velocity and parking_distance from the command line 
        self.add_on_set_parameters_callback(self.parameters_callback)

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()

        # Current angle of the cone relative to front of the car
        angle_error = np.arctan2(self.relative_y, self.relative_x)

        # Current distance of the car from the cone
        distance = np.sqrt(self.relative_x**2 + self.relative_y**2)

        distance_error = distance - self.parking_distance

        x_error = self.relative_x - self.parking_distance

        high_angular_error = (np.abs(angle_error) > self.angular_error_threshold)

        # Some PID setup
        current_time = self.get_clock().now()
        dt = (current_time.nanoseconds - self.previous_time.nanoseconds) / (10**9)

        # Car is pointed in the wrong direction
        # Reversing for 3 point parking type maneuvers
        heading = -angle_error if self.reverse else angle_error

        if(distance_error <= self.distance_error_threshold[0]):
                self.reverse = True
        elif distance_error >= self.distance_error_threshold[1]:
                self.reverse = False
        
        
        if not high_angular_error and distance_error > self.distance_error_threshold[0] and distance_error < self.distance_error_threshold[1]:
            velocity = self.parking_velocity * ((x_error)/self.parking_distance)
            heading /= 2        
        else:
            velocity = -self.parking_velocity if self.reverse else self.parking_velocity

        self.integrated_dist_error += max(min(dt * self.previous_x_error, self.integral_max), self.integral_min)
        self.previous_x_error = x_error
        self.previous_time = current_time
        



        drive_cmd.header.frame_id = "base_link"
        drive_cmd.header.stamp = current_time.to_msg()
        drive_cmd.drive.steering_angle = heading
        drive_cmd.drive.speed = velocity
        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        error_msg.x_error = self.relative_x - self.parking_distance
        error_msg.y_error =  self.relative_y
        error_msg.distance_error = np.hypot(error_msg.x_error, error_msg.y_error)
        
        self.error_pub.publish(error_msg)

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'parking_velocity':
                self.parking_velocity = param.value
                self.get_logger().info(f"Updated parking velocity to {self.parking_velocity}")
            elif param.name == 'parking_distance':
                self.parking_distance = param.value
                self.get_logger().info(f"Updated parking distance to {self.parking_distance}")
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()