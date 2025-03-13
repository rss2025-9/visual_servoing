#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import String
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped

class PixelExtractor(Node):
    def __init__(self):
        super().__init__("pixel_extractor")
        self.subscription = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color/image", self.image_callback, 10)
        self.frame = "Map"

    def image_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imshow("Frame", self.frame)
        cv2.setMouseCallback("Frame", self.click_event)
        cv2.waitKey(1)

    def click_event(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Pixel Coordinates: (u={x}, v={y})")

def main(args=None):
    rclpy.init(args=args)
    pixel_extractor = PixelExtractor()
    rclpy.spin(pixel_extractor)
    rclpy.shutdown()

if __name__ == "__main__":
    main()