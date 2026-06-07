#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node

from geometry_msgs.msg import Twist, Pose2D
from std_msgs.msg import Bool, String, Float32
from rclpy.qos import qos_profile_sensor_data


class PuzzlebotVPFController(Node):

    def __init__(self):
        super().__init__("puzzlebot_vpf_controller")
        self.get_logger().info("VPF controller with smooth locked avoidance started")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.goal_pub = self.create_publisher(Bool, "/goal_reached", 10)

        self.create_subscription(Pose2D, "/odom", self.cb_odom, 10)
        self.create_subscription(Pose2D, "/goals", self.cb_goal, 10)

        self.create_subscription(Bool, "/obstacle_detected", self.cb_obstacle_detected, qos_profile_sensor_data)
        self.create_subscription(String, "/obstacle_zone", self.cb_obstacle_zone, qos_profile_sensor_data)
        self.create_subscription(Float32, "/obstacle_error", self.cb_obstacle_error, qos_profile_sensor_data)
        self.create_subscription(Float32, "/obstacle_area", self.cb_obstacle_area, qos_profile_sensor_data)
        self.create_subscription(Float32, "/obstacle_y", self.cb_obstacle_y, qos_profile_sensor_data)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.xt = 0.0
        self.yt = 0.0
        self.got_goal = False

        self.obstacle_detected = False
        self.obstacle_zone = "none"
        self.obstacle_error = 0.0
        self.obstacle_area = 0.0
        self.obstacle_y = 0.0

        self.alpha_filter = 0.25

        # Control hacia meta
        self.k_att = 1.0
        self.k_w = 1.8

        # Detección de obstáculo
        self.rep_area_min = 10000.0
        self.rep_y_min = 0.10

        # Velocidades
        self.v_max = 0.09
        self.w_max = 0.55
        self.v_avoid = 0.055

        self.goal_tolerance = 0.08

        # Evasión bloqueada
        self.avoiding = False
        self.avoid_dir = 0.0
        self.avoid_counter = 0

        # Timer = 0.1 s
        self.min_avoid_steps = 35      # mínimo 3.5 s rodeando
        self.avoid_lock_steps = 75     # máximo 7.5 s evadiendo
        self.clear_required = 20       # 2.0 s viendo libre
        self.clear_counter = 0

        self.create_timer(0.1, self.control_loop)

    def cb_odom(self, msg):
        self.x = msg.x
        self.y = msg.y
        self.theta = msg.theta

    def cb_goal(self, msg):
        self.xt = msg.x
        self.yt = msg.y
        self.got_goal = True
        self.get_logger().info(f"Nueva meta: ({self.xt:.2f}, {self.yt:.2f})")

    def cb_obstacle_detected(self, msg):
        self.obstacle_detected = msg.data

    def cb_obstacle_zone(self, msg):
        self.obstacle_zone = msg.data

    def cb_obstacle_error(self, msg):
        self.obstacle_error = (
            self.alpha_filter * msg.data
            + (1.0 - self.alpha_filter) * self.obstacle_error
        )

    def cb_obstacle_area(self, msg):
        self.obstacle_area = (
            self.alpha_filter * msg.data
            + (1.0 - self.alpha_filter) * self.obstacle_area
        )

    def cb_obstacle_y(self, msg):
        self.obstacle_y = (
            self.alpha_filter * msg.data
            + (1.0 - self.alpha_filter) * self.obstacle_y
        )

    def control_loop(self):

        if not self.got_goal:
            self.stop()
            return

        dx = self.xt - self.x
        dy = self.yt - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < self.goal_tolerance:
            self.stop()
            self.got_goal = False

            msg = Bool()
            msg.data = True
            self.goal_pub.publish(msg)

            self.reset_avoidance()
            self.get_logger().info("Meta alcanzada")
            return

        if self.avoiding:
            self.smooth_avoid_motion()
            return

        if self.is_obstacle_relevant():
            self.start_avoidance()
            self.smooth_avoid_motion()
            return

        self.go_to_goal(dx, dy, dist)

    def go_to_goal(self, dx, dy, dist):

        desired_theta = math.atan2(dy, dx)

        error_theta = math.atan2(
            math.sin(desired_theta - self.theta),
            math.cos(desired_theta - self.theta)
        )

        alignment = max(0.0, math.cos(error_theta))

        v = self.v_max * alignment
        w = self.k_w * error_theta

        v = max(min(v, self.v_max), 0.0)
        w = max(min(w, self.w_max), -self.w_max)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"NORMAL | dist={dist:.2f} v={v:.2f} w={w:.2f}",
            throttle_duration_sec=1.0
        )

    def is_obstacle_relevant(self):
        return (
            self.obstacle_detected
            and (
                self.obstacle_area > self.rep_area_min
                or self.obstacle_y > self.rep_y_min
            )
        )

    def start_avoidance(self):

        self.avoiding = True
        self.avoid_counter = 0
        self.clear_counter = 0

        if self.obstacle_error >= 0:
            self.avoid_dir = 1.0
        else:
            self.avoid_dir = -1.0

        self.get_logger().warn(
            f"EVASIÓN iniciada | area={self.obstacle_area:.0f} "
            f"y={self.obstacle_y:.2f} error={self.obstacle_error:.2f} "
            f"dir={self.avoid_dir}"
        )

    def smooth_avoid_motion(self):

        cmd = Twist()
        cmd.linear.x = self.v_avoid
        cmd.angular.z = 0.32 * self.avoid_dir
        self.cmd_pub.publish(cmd)

        self.avoid_counter += 1

        obstacle_cleared = (
            (not self.obstacle_detected)
            or (
                self.obstacle_area < 8000.0
                and self.obstacle_y < 0.10
            )
        )

        # No puede terminar evasión antes del tiempo mínimo.
        if self.avoid_counter >= self.min_avoid_steps:
            if obstacle_cleared:
                self.clear_counter += 1
            else:
                self.clear_counter = 0
        else:
            self.clear_counter = 0

        if (
            self.clear_counter >= self.clear_required
            or self.avoid_counter >= self.avoid_lock_steps
        ):
            self.reset_avoidance()
            self.get_logger().info("EVASIÓN terminada → regresando a la meta")

        self.get_logger().info(
            f"AVOID | v={cmd.linear.x:.2f} w={cmd.angular.z:.2f} "
            f"area={self.obstacle_area:.0f} y={self.obstacle_y:.2f} "
            f"counter={self.avoid_counter}",
            throttle_duration_sec=1.0
        )

    def reset_avoidance(self):
        self.avoiding = False
        self.avoid_dir = 0.0
        self.avoid_counter = 0
        self.clear_counter = 0

    def stop(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotVPFController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
