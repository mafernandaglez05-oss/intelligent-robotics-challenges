
#!/usr/bin/env python3
"""
sugeno_controller.py  —  FIS Sugeno, NumPy puro, sin skfuzzy
─────────────────────────────────────────────────────────────
Adaptado al nodo de visión LineDetector:
  · /line_error  publica error NORMALIZADO [-1.0, 1.0]  (no píxeles)
  · QoS: qos_profile_sensor_data  (best effort, volatile)
  · /line_lost   Bool — para detener el robot si se pierde la línea
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy import qos
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist


# ══════════════════════════════════════════════════════════════════════════════
#  FUZZY HELPERS — NumPy puro
# ══════════════════════════════════════════════════════════════════════════════

def _gaussmf(x: float, mean: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((x - mean) / sigma) ** 2))


# ══════════════════════════════════════════════════════════════════════════════
#  NODO
# ══════════════════════════════════════════════════════════════════════════════

class SugenoController(Node):

    _DEFAULTS = {
        'v_normal':   0.07,
        'v_slow':     0.05,
        'v_min':      0.045,
        'w_max':      1.3,
        'sigma_wide': 0.28,   # en unidades normalizadas (≈90px/320)
        'sigma_zero': 0.16,   # MF central más estrecha  (≈50px/320)
    }

    def __init__(self):
        super().__init__('sugeno_controller')

        for name, val in self._DEFAULTS.items():
            self.declare_parameter(name, val)

        self.v_normal = self.get_parameter('v_normal').value
        self.v_slow   = self.get_parameter('v_slow').value
        self.v_min    = self.get_parameter('v_min').value
        self.w_max    = self.get_parameter('w_max').value
        sw            = self.get_parameter('sigma_wide').value
        sz            = self.get_parameter('sigma_zero').value

        # Centros de MFs gaussianas en universo normalizado [-1, 1]
        self._centers = {
            'NL': -0.875,
            'NM': -0.406,
            'Z':   0.0,
            'PM':  0.406,
            'PL':  0.875,
        }
        self._sigmas = {k: (sz if k == 'Z' else sw) for k in self._centers}

        # Singletons de salida
        wm = self.w_max
        self._V_s = {
            'NL': self.v_min,
            'NM': self.v_slow,
            'Z':  self.v_normal,
            'PM': self.v_slow,
            'PL': self.v_min,
        }
        self._W_s = {
            'NL': -wm,
            'NM': -wm * 0.46,
            'Z':   0.0,
            'PM':  wm * 0.46,
            'PL':  wm,
        }

        # Estado
        self._line_lost = False

        # ── Suscriptores — mismo QoS que LineDetector ────────────────────────
        self.create_subscription(
            Float32, '/line_error',
            self._error_cb,
            qos.qos_profile_sensor_data)

        self.create_subscription(
            Bool, '/line_lost',
            self._lost_cb,
            qos.qos_profile_sensor_data)

        # ── Publicador ───────────────────────────────────────────────────────
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info(
            f'[Sugeno] Listo | error normalizado [-1,1] | '
            f'V=[{self.v_min},{self.v_normal}] | ω=±{self.w_max}')

    # ── Inferencia Sugeno — weighted average ─────────────────────────────────
    def _infer(self, e: float):
        firing = {
            k: _gaussmf(e, self._centers[k], self._sigmas[k])
            for k in self._centers
        }
        w_sum = sum(firing.values())
        if w_sum < 1e-9:
            return self.v_min, 0.0

        v_out = sum(firing[k] * self._V_s[k] for k in firing) / w_sum
        w_out = sum(firing[k] * self._W_s[k] for k in firing) / w_sum
        return v_out, w_out

    # ── Callback /line_lost ──────────────────────────────────────────────────
    def _lost_cb(self, msg: Bool):
        self._line_lost = msg.data
        if self._line_lost:
            self.pub.publish(Twist())
            self.get_logger().warn('[Sugeno] Línea perdida — robot detenido')

    # ── Callback /line_error ─────────────────────────────────────────────────
    def _error_cb(self, msg: Float32):
        if self._line_lost:
            return

        e = float(np.clip(msg.data, -1.0, 1.0))

        try:
            v_cmd, w_cmd = self._infer(e)
        except Exception as ex:
            self.get_logger().warn(f'FIS error: {ex}')
            v_cmd, w_cmd = self.v_min, 0.0

        twist = Twist()
        twist.linear.x  = float(np.clip(v_cmd, self.v_min, self.v_normal))
        twist.angular.z = float(np.clip(w_cmd, -self.w_max, self.w_max))
        self.pub.publish(twist)

        self.get_logger().debug(
            f'e={e:+.3f} → V={twist.linear.x:.4f}  ω={twist.angular.z:+.4f}')


# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = SugenoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
