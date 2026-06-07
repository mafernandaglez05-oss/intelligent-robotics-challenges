#!/usr/bin/env python3


import rclpy
import time
import math
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32
from rclpy.qos import qos_profile_sensor_data


class PuzzlebotOdometry(Node):

    def __init__(self):
        super().__init__('odometry')
        self.get_logger().info("Odometry node iniciado (modo TurtleSim → /odom Pose2D)")

        #  Parámetros físicos del robot 
        self.r    = 0.05    # radio de rueda [m]
        self.L    = 0.182   # distancia entre ruedas (baseline) [m]
        self.rate = 100     # frecuencia de integración [Hz]

        #  Encoder 
        self.wR    = 0.0
        self.wL    = 0.0
        self.v     = 0.0
        self.w     = 0.0
        self.x     = 5.5    # posición inicial en TurtleSim [m]
        self.y     = 5.5
        self.theta = 0.0

        #  Publisher 
        self.pub = self.create_publisher(Pose2D, '/odom', 10)

        #  Suscriptor  TurtleSim 
        self.create_subscription(Pose, '/turtle1/pose', self.cb_pose, 10)

        #  Suscriptores  robot físico 
        # self.create_subscription(
        #     Float32, '/VelocityEncR', self.cb_wR, qos_profile_sensor_data)
        # self.create_subscription(
        #     Float32, '/VelocityEncL', self.cb_wL, qos_profile_sensor_data)
        # self.create_timer(1.0 / self.rate, self.cb_odometry)

        self.t0 = time.time()



    def cb_pose(self, msg: Pose):
     
        out       = Pose2D()
        out.x     = msg.x
        out.y     = msg.y
        out.theta = msg.theta
        self.pub.publish(out)



    def cb_wR(self, msg: Float32):
        self.wR = msg.data

    def cb_wL(self, msg: Float32):
        self.wL = msg.data

    #  Odometría robot 

    def cb_odometry(self):

        dt      = time.time() - self.t0
        self.t0 = time.time()

        self.v = self.r * (self.wR + self.wL) / 2.0
        self.w = self.r * (self.wR - self.wL) / self.L

        self.x     += dt * self.v * math.cos(self.theta)
        self.y     += dt * self.v * math.sin(self.theta)
        self.theta += dt * self.w
        self.theta  = math.atan2(math.sin(self.theta), math.cos(self.theta))

        msg       = Pose2D()
        msg.x     = self.x
        msg.y     = self.y
        msg.theta = self.theta
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
