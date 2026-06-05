#!/usr/bin/env python3
"""
Odometría para TurtleSim / Robot Físico (ROS2 Humble)
======================================================
Modo TurtleSim (activo):
  Suscribe a /turtle1/pose (turtlesim/msg/Pose) y republica como
  geometry_msgs/msg/Pose2D en /odom. Las coordenadas se pasan sin
  offset para que coincidan exactamente con los waypoints del planner A*.

Modo robot físico (comentado):
  Integra odometría diferencial usando velocidades de encoder:
    v     = r * (wR + wL) / 2        [velocidad lineal]
    w     = r * (wR - wL) / L        [velocidad angular]
    x    += dt * v * cos(theta)      [integración Euler]
    y    += dt * v * sin(theta)
    theta += dt * w
  donde r = radio de rueda, L = distancia entre ruedas.

Tópicos (modo TurtleSim):
  Suscribe → /turtle1/pose   (turtlesim/msg/Pose)
  Publica  → /odom           (geometry_msgs/msg/Pose2D)

Tópicos adicionales (modo robot físico, descomentar):
  Suscribe → /VelocityEncR   (std_msgs/msg/Float32)
  Suscribe → /VelocityEncL   (std_msgs/msg/Float32)
"""

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

        # ── Parámetros físicos del robot (modo encoder) ──────────────────────
        self.r    = 0.05    # radio de rueda [m]
        self.L    = 0.182   # distancia entre ruedas (baseline) [m]
        self.rate = 100     # frecuencia de integración [Hz]

        # ── Estado encoder (modo robot físico) ──────────────────────────────
        self.wR    = 0.0
        self.wL    = 0.0
        self.v     = 0.0
        self.w     = 0.0
        self.x     = 5.5    # posición inicial en TurtleSim [m]
        self.y     = 5.5
        self.theta = 0.0

        # ── Publisher ────────────────────────────────────────────────────────
        self.pub = self.create_publisher(Pose2D, '/odom', 10)

        # ── Suscriptor modo TurtleSim (activo) ──────────────────────────────
        self.create_subscription(Pose, '/turtle1/pose', self.cb_pose, 10)

        # ── Suscriptores modo robot físico (descomentar para usar) ───────────
        # self.create_subscription(
        #     Float32, '/VelocityEncR', self.cb_wR, qos_profile_sensor_data)
        # self.create_subscription(
        #     Float32, '/VelocityEncL', self.cb_wL, qos_profile_sensor_data)
        # self.create_timer(1.0 / self.rate, self.cb_odometry)

        self.t0 = time.time()

    # ── Callback TurtleSim ───────────────────────────────────────────────────

    def cb_pose(self, msg: Pose):
        """
        Reenvía la pose de turtlesim directamente como Pose2D.
        No se aplica offset: las coordenadas absolutas de TurtleSim (~0-11 m)
        deben coincidir exactamente con los waypoints del planner A*.
        """
        out       = Pose2D()
        out.x     = msg.x
        out.y     = msg.y
        out.theta = msg.theta
        self.pub.publish(out)

    # ── Callbacks encoder (robot físico) ────────────────────────────────────

    def cb_wR(self, msg: Float32):
        self.wR = msg.data

    def cb_wL(self, msg: Float32):
        self.wL = msg.data

    # ── Odometría diferencial (robot físico) ─────────────────────────────────

    def cb_odometry(self):
        """
        Integración de odometría diferencial por encoders.

        Modelo cinemático differential drive:
          v      = r * (wR + wL) / 2    → velocidad lineal del centro
          w      = r * (wR - wL) / L    → velocidad angular

        Integración Euler (primer orden):
          x(t+dt)     = x(t)     + dt * v * cos(theta(t))
          y(t+dt)     = y(t)     + dt * v * sin(theta(t))
          theta(t+dt) = theta(t) + dt * w

        El ángulo se normaliza a [-π, π] con atan2(sin, cos) para evitar
        acumulación de error y wrap-around.
        """
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
