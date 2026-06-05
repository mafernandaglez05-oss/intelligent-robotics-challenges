#!/usr/bin/env python3
"""
mamdani_controller.py
─────────────────────
Nodo ROS 2 — Controlador difuso Mamdani para seguimiento de línea (Puzzlebot).

Suscribe : /line_error   (std_msgs/Float32)  — error en píxeles, centro=0
Publica  : /cmd_vel      (geometry_msgs/Twist)

Parámetros tuneables (ros2 param set):
  ~v_normal   : velocidad lineal máxima  [default 0.07]
  ~v_slow     : velocidad en curva       [default 0.05]
  ~v_min      : velocidad mínima         [default 0.045]
  ~w_max      : velocidad angular máx    [default 1.3]
  ~img_width  : ancho imagen en px       [default 640]
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import warnings
warnings.filterwarnings('ignore')


class MamdaniController(Node):

    def __init__(self):
        super().__init__('mamdani_controller')

        # ── Parámetros ──────────────────────────────────────────────────────
        self.declare_parameter('v_normal',  0.07)
        self.declare_parameter('v_slow',    0.05)
        self.declare_parameter('v_min',     0.045)
        self.declare_parameter('w_max',     1.3)
        self.declare_parameter('img_width', 640)

        self.v_normal  = self.get_parameter('v_normal').value
        self.v_slow    = self.get_parameter('v_slow').value
        self.v_min     = self.get_parameter('v_min').value
        self.w_max     = self.get_parameter('w_max').value
        self.img_width = self.get_parameter('img_width').value

        self.e_max = self.img_width // 2  # 320 px

        # ── Construir FIS ───────────────────────────────────────────────────
        self._build_fis()
        self.get_logger().info('Mamdani FIS construido ✓')

        # ── ROS 2 interfaces ────────────────────────────────────────────────
        self.sub = self.create_subscription(
            Float32, '/line_error', self.error_callback, 10)

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info(
            f'MamdaniController listo | e_range=[{-self.e_max}, {self.e_max}] px')

    # ────────────────────────────────────────────────────────────────────────
    def _build_fis(self):
        """Construye el sistema de inferencia Mamdani con skfuzzy."""
        emax = float(self.e_max)

        # Universos
        error = ctrl.Antecedent(np.linspace(-emax, emax, int(2*emax)+1), 'error')
        V     = ctrl.Consequent(np.linspace(self.v_min, self.v_normal, 200), 'V')
        omega = ctrl.Consequent(np.linspace(-self.w_max, self.w_max, 200), 'omega')

        # ── MFs del error (5 triangulares) ──────────────────────────────────
        error['NL'] = fuzz.trimf(error.universe, [-emax,  -emax,  -emax*0.375])
        error['NM'] = fuzz.trimf(error.universe, [-emax*0.75, -emax*0.25, 0.0])
        error['Z']  = fuzz.trimf(error.universe, [-emax*0.19, 0.0, emax*0.19])
        error['PM'] = fuzz.trimf(error.universe, [0.0, emax*0.25, emax*0.75])
        error['PL'] = fuzz.trimf(error.universe, [emax*0.375, emax, emax])

        # ── MFs de V (5) ────────────────────────────────────────────────────
        v_range = self.v_normal - self.v_min
        V['very_slow'] = fuzz.trimf(V.universe,
            [self.v_min, self.v_min, self.v_min + 0.2*v_range])
        V['slow']      = fuzz.trimf(V.universe,
            [self.v_min, self.v_slow, self.v_min + 0.55*v_range])
        V['medium']    = fuzz.trimf(V.universe,
            [self.v_min + 0.2*v_range, self.v_min + 0.5*v_range, self.v_min + 0.75*v_range])
        V['fast']      = fuzz.trimf(V.universe,
            [self.v_min + 0.5*v_range, self.v_min + 0.75*v_range, self.v_normal])
        V['normal']    = fuzz.trimf(V.universe,
            [self.v_min + 0.7*v_range, self.v_normal, self.v_normal])

        # ── MFs de omega (5) ────────────────────────────────────────────────
        wmax = self.w_max
        omega['NL'] = fuzz.trimf(omega.universe, [-wmax, -wmax, -wmax*0.38])
        omega['NM'] = fuzz.trimf(omega.universe, [-wmax, -wmax*0.46, 0.0])
        omega['Z']  = fuzz.trimf(omega.universe, [-wmax*0.15, 0.0, wmax*0.15])
        omega['PM'] = fuzz.trimf(omega.universe, [0.0, wmax*0.46, wmax])
        omega['PL'] = fuzz.trimf(omega.universe, [wmax*0.38, wmax, wmax])

        # ── Reglas ──────────────────────────────────────────────────────────
        # Error negativo → línea a la IZQUIERDA → girar izquierda (ω negativo)
        # Error positivo → línea a la DERECHA  → girar derecha   (ω positivo)
        rules = [
            ctrl.Rule(error['NL'], (V['very_slow'], omega['NL'])),
            ctrl.Rule(error['NM'], (V['slow'],      omega['NM'])),
            ctrl.Rule(error['Z'],  (V['normal'],    omega['Z'])),
            ctrl.Rule(error['PM'], (V['slow'],      omega['PM'])),
            ctrl.Rule(error['PL'], (V['very_slow'], omega['PL'])),
        ]

        fis = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(fis)

    # ────────────────────────────────────────────────────────────────────────
    def error_callback(self, msg: Float32):
        error_px = float(msg.data)

        # Clip por seguridad
        error_px = float(np.clip(error_px, -self.e_max, self.e_max))

        try:
            self.sim.input['error'] = error_px
            self.sim.compute()
            v_cmd     = float(self.sim.output['V'])
            omega_cmd = float(self.sim.output['omega'])
        except Exception as ex:
            self.get_logger().warn(f'FIS error: {ex} — usando defaults')
            v_cmd     = self.v_min
            omega_cmd = 0.0

        twist = Twist()
        twist.linear.x  = float(np.clip(v_cmd,     self.v_min, self.v_normal))
        twist.angular.z = float(np.clip(omega_cmd, -self.w_max, self.w_max))
        self.pub.publish(twist)

        self.get_logger().debug(
            f'error={error_px:+.1f}px  V={twist.linear.x:.4f}  ω={twist.angular.z:+.4f}')


# ────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = MamdaniController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
