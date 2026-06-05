#!/usr/bin/env python3
"""
Puzzlebot Controller — Closed-Loop para TurtleSim (ROS2 Humble)
================================================================
Máquina de estados: stop → align → move → stop

Tópicos:
  Publica  → /turtle1/cmd_vel  (geometry_msgs/msg/Twist)   comandos de velocidad
  Publica  → /goal_reached     (std_msgs/msg/Bool)         confirmación de waypoint
  Suscribe → /odom             (geometry_msgs/msg/Pose2D)  posición actual
  Suscribe → /goals            (turtlesim/msg/Pose)        waypoint objetivo

Parámetros:
  Kw              (float, default 4.0)   — ganancia del controlador angular
  v_normal        (float, default 0.2)   — velocidad lineal nominal [m/s]
  tolerance_dist  (float, default 0.05)  — tolerancia de llegada [m]
  tolerance_angle (float, default 0.05)  — tolerancia de alineación [rad]
"""

import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from turtlesim.msg import Pose
from std_msgs.msg import Bool


class PuzzlebotControllerClass(Node):

    def __init__(self):
        super().__init__("controller")
        self.get_logger().info("Puzzlebot controller iniciado")

        # ── Publishers ──────────────────────────────────────────────────────
        self.pub        = self.create_publisher(Twist, "/turtle1/cmd_vel", 1)
        self.status_pub = self.create_publisher(Bool,  "/goal_reached",    10)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.create_subscription(Pose2D, "/odom",  self.cb_pose,   1)
        self.create_subscription(Pose,   "/goals", self.cb_target, 1)

        # ── Parámetros ──────────────────────────────────────────────────────
        self.declare_parameter("Kw",              4.0)
        self.declare_parameter("v_normal",        0.2)
        self.declare_parameter("tolerance_dist",  0.05)
        self.declare_parameter("tolerance_angle", 0.05)

        self.Kw              = self.get_parameter("Kw").value
        self.v_normal        = self.get_parameter("v_normal").value
        self.tolerance_dist  = self.get_parameter("tolerance_dist").value
        self.tolerance_angle = self.get_parameter("tolerance_angle").value

        # ── FSM ─────────────────────────────────────────────────────────────
        self.state         = "stop"
        self.end_of_action = False
        self.got_target    = False

        # ── Odometría ───────────────────────────────────────────────────────
        self.x, self.y, self.theta = 0.0, 0.0, 0.0

        # ── Meta actual ─────────────────────────────────────────────────────
        self.xt, self.yt = 0.0, 0.0

        self.create_timer(0.1, self.state_machine)

    # ── Callbacks ───────────────────────────────────────────────────────────

    def cb_target(self, msg: Pose):
        self.xt = msg.x
        self.yt = msg.y
        self.got_target    = True
        self.end_of_action = False
        self.get_logger().info(f"Nueva meta: ({self.xt:.2f}, {self.yt:.2f})")

    def cb_pose(self, msg: Pose2D):
        self.x     = msg.x
        self.y     = msg.y
        self.theta = msg.theta

    # ── Máquina de estados ───────────────────────────────────────────────────

    def state_machine(self):
        if self.state == "stop":
            self.stop()
            if self.got_target:
                self.state = "align"

        elif self.state == "align":
            self.go_to_angle()
            if self.end_of_action:
                self.state         = "move"
                self.end_of_action = False

        elif self.state == "move":
            self.go_to_point()
            if self.end_of_action:
                status_msg      = Bool()
                status_msg.data = True
                self.status_pub.publish(status_msg)
                self.state         = "stop"
                self.got_target    = False
                self.end_of_action = False

    # ── Acciones ─────────────────────────────────────────────────────────────

    def stop(self):
        self.pub.publish(Twist())

    def go_to_angle(self):
        """
        Gira el robot en su lugar hasta apuntar hacia el waypoint.
        Usa atan2 para obtener el ángulo deseado y normaliza el error
        al rango [-π, π] para evitar giros innecesarios de 360°.
        """
        Dx = self.xt - self.x
        Dy = self.yt - self.y
        target_angle = math.atan2(Dy, Dx)
        etheta = math.atan2(
            math.sin(target_angle - self.theta),
            math.cos(target_angle - self.theta)
        )

        msg = Twist()
        if abs(etheta) > self.tolerance_angle:
            w_cmd = self.Kw * etheta
            msg.angular.z = max(min(w_cmd, 0.7), -0.7)
            self.pub.publish(msg)
        else:
            self.stop()
            self.end_of_action = True
            self.get_logger().info("Ángulo alineado ✓")

    def go_to_point(self):
        """
        Avanza hacia el waypoint con corrección angular proporcional.
        El robot differential drive no puede moverse lateralmente,
        por eso se alinea primero y luego avanza con corrección continua.

        Cinemática differential drive:
          v = (r/2) * (wR + wL)   — velocidad lineal
          w = (r/L) * (wR - wL)   — velocidad angular
        En el controlador: v se fija en v_normal y w = Kw * error_angular.
        """
        Dx = self.xt - self.x
        Dy = self.yt - self.y
        dist = math.sqrt(Dx**2 + Dy**2)

        target_angle = math.atan2(Dy, Dx)
        etheta = math.atan2(
            math.sin(target_angle - self.theta),
            math.cos(target_angle - self.theta)
        )

        msg = Twist()
        if dist > self.tolerance_dist:
            msg.linear.x  = self.v_normal
            w_cmd         = self.Kw * etheta
            msg.angular.z = max(min(w_cmd, 1.0), -1.0)
            self.pub.publish(msg)
        else:
            self.stop()
            self.end_of_action = True
            self.get_logger().info(f"Waypoint alcanzado ✓  (dist={dist:.3f} m)")


def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotControllerClass()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
