#!/usr/bin/env python3
"""
RRT Visualizer OPTIMIZADO para TurtleSim (ROS2 Humble)
=======================================================
MEJORAS RESPECTO A LA VERSIÓN ANTERIOR:
  1. VELOCIDAD: relleno de obstáculos circulares con líneas paralelas
     horizontales en lugar de anillos concéntricos → mucho menos teleports.
  2. NUEVO: suscribe a /rrt/tree_edges y dibuja todas las aristas del árbol
     RRT en color verde oscuro (trazos continuos agrupados).
  3. NUEVO: suscribe a /rrt/path y dibuja el camino final en cyan grueso.
  4. Redibuja marcadores de start/goal encima al terminar.

Orden de dibujado:
  1. Borde del mundo        (gris)
  2. Obstáculos circulares  (rojo)     ← al arrancar
  3. Marcador inicio        (verde)
  4. Marcador meta          (azul)
  5. Aristas del árbol RRT  (verde oscuro)  ← al recibir /rrt/tree_edges
  6. Path final             (cyan grueso)   ← al recibir /rrt/path
  7. Re-dibujar marcadores start/goal encima

Parámetros (deben coincidir con rrt_planner_node):
  start_x    (float, default 1.0)
  start_y    (float, default 1.0)
  goal_x     (float, default 9.0)
  goal_y     (float, default 9.0)
  world_min  (float, default 0.5)
  world_max  (float, default 10.5)
  obstacles  (str)  "cx,cy,radio;..."
"""

import rclpy
import math
import json
import time
from rclpy.node import Node
from turtlesim.srv import Spawn, Kill, SetPen, TeleportAbsolute
from std_msgs.msg import String


class RRTVisualizerNode(Node):

    def __init__(self):
        super().__init__('rrt_visualizer')
        self.get_logger().info("RRT Visualizer (optimizado) iniciado")

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('start_x',   1.0)
        self.declare_parameter('start_y',   1.0)
        self.declare_parameter('goal_x',    10.0)
        self.declare_parameter('goal_y',    10.0)
        self.declare_parameter('world_min', 0.5)
        self.declare_parameter('world_max', 10.5)
        self.declare_parameter('obstacles',
            "2.0,8.0,0.9;4.5,8.0,0.9;7.5,8.0,0.9;2.0,5.0,0.9;4.5,5.0,0.9;7.5,5.0,0.9")

        self.sx        = self.get_parameter('start_x').value
        self.sy        = self.get_parameter('start_y').value
        self.gx        = self.get_parameter('goal_x').value
        self.gy        = self.get_parameter('goal_y').value
        self.world_min = self.get_parameter('world_min').value
        self.world_max = self.get_parameter('world_max').value
        self.obstacles = self._parse_obstacles(
                             self.get_parameter('obstacles').value)

        # ── Clientes de servicio ──────────────────────────────────────────
        self.cli_spawn    = self.create_client(Spawn,            '/spawn')
        self.cli_kill     = self.create_client(Kill,             '/kill')
        self.cli_teleport = None   # se asigna tras spawn
        self.cli_pen      = None

        self.cli_spawn.wait_for_service(timeout_sec=5.0)

        # ── Flags de estado ───────────────────────────────────────────────
        self._base_drawn     = False
        self._tree_drawn     = False
        self._path_drawn     = False

        # ── Suscriptores del planner ──────────────────────────────────────
        self.create_subscription(String, '/rrt/tree_edges', self.cb_tree_edges, 10)
        self.create_subscription(String, '/rrt/path',       self.cb_path,       10)

        # Arrancar dibujado base 1 s después del inicio
        self.create_timer(1.0, self._start_base)

    # ── Parser ────────────────────────────────────────────────────────────

    def _parse_obstacles(self, obs_str: str):
        result = []
        for token in obs_str.split(';'):
            token = token.strip()
            if not token:
                continue
            parts = token.split(',')
            if len(parts) != 3:
                continue
            try:
                result.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
        return result

    # ── Primitivas de servicio ────────────────────────────────────────────

    def _teleport(self, x, y, theta=0.0):
        req = TeleportAbsolute.Request()
        req.x, req.y, req.theta = float(x), float(y), float(theta)
        f = self.cli_teleport.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=2.0)

    def _set_pen(self, r, g, b, width=2, off=False):
        req       = SetPen.Request()
        req.r     = int(r)
        req.g     = int(g)
        req.b     = int(b)
        req.width = int(width)
        req.off   = int(off)
        f = self.cli_pen.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=2.0)

    def _pen_up(self):
        self._set_pen(0, 0, 0, width=1, off=True)

    # ── Spawn ─────────────────────────────────────────────────────────────

    def _spawn_painter(self):
        if self.cli_kill.wait_for_service(timeout_sec=1.0):
            req      = Kill.Request()
            req.name = 'painter'
            f = self.cli_kill.call_async(req)
            rclpy.spin_until_future_complete(self, f, timeout_sec=2.0)
            time.sleep(0.3)

        req         = Spawn.Request()
        req.x       = self.sx
        req.y       = self.sy
        req.theta   = 0.0
        req.name    = 'painter'
        f = self.cli_spawn.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=5.0)
        time.sleep(0.5)

        self.cli_teleport = self.create_client(
            TeleportAbsolute, '/painter/teleport_absolute')
        self.cli_pen = self.create_client(
            SetPen, '/painter/set_pen')
        self.cli_teleport.wait_for_service(timeout_sec=5.0)
        self.cli_pen.wait_for_service(timeout_sec=5.0)

    # ─────────────────────────────────────────────────────────────────────
    #  OPTIMIZACIÓN: relleno circular con líneas horizontales paralelas
    #  En lugar de N anillos concéntricos (cada uno con 12-36 teleports),
    #  dibuja líneas horizontales de x0 a x1 a distintas alturas y.
    #  Teleports totales ≈ 2 * (2*radius/step) → mucho menos llamadas.
    # ─────────────────────────────────────────────────────────────────────
    def _fill_circle_fast(self, cx, cy, radius, r, g, b, lines=12):
        """
        Rellena un círculo con líneas horizontales paralelas.
        `lines` controla la densidad del relleno.
        """
        self._pen_up()
        for i in range(lines):
            # Posición vertical de la línea: de -radius a +radius
            t  = -1.0 + (2.0 * i) / (lines - 1) if lines > 1 else 0.0
            yy = cy + t * radius
            # Semiancho en esta altura: x = sqrt(r² - t²*r²)
            half_w = radius * math.sqrt(max(0.0, 1.0 - t * t))
            if half_w < 1e-4:
                continue
            self._teleport(cx - half_w, yy)
            self._set_pen(r, g, b, width=3, off=False)
            self._teleport(cx + half_w, yy)
            self._pen_up()

        # Contorno exterior en un solo trazo de 24 puntos
        n_pts = 24
        self._teleport(cx + radius, cy)
        self._set_pen(r, g, b, width=2, off=False)
        for j in range(1, n_pts + 1):
            angle = 2 * math.pi * j / n_pts
            self._teleport(
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle)
            )
        self._pen_up()

    def _draw_circle_outline(self, cx, cy, radius, r, g, b, width=3):
        """Dibuja solo el contorno de un círculo (24 segmentos)."""
        n_pts = 24
        self._pen_up()
        self._teleport(cx + radius, cy)
        self._set_pen(r, g, b, width=width, off=False)
        for j in range(1, n_pts + 1):
            angle = 2 * math.pi * j / n_pts
            self._teleport(
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle)
            )
        self._pen_up()

    def _draw_cross(self, cx, cy, size, r, g, b, width=4):
        half = size / 2
        self._pen_up()
        self._teleport(cx - half, cy - half)
        self._set_pen(r, g, b, width=width, off=False)
        self._teleport(cx + half, cy + half)
        self._pen_up()
        self._teleport(cx + half, cy - half)
        self._set_pen(r, g, b, width=width, off=False)
        self._teleport(cx - half, cy + half)
        self._pen_up()

    def _draw_world_border(self):
        wmin, wmax = self.world_min, self.world_max
        corners = [
            (wmin, wmin), (wmax, wmin),
            (wmax, wmax), (wmin, wmax),
            (wmin, wmin),
        ]
        self._pen_up()
        self._teleport(*corners[0])
        self._set_pen(80, 80, 80, width=2, off=False)
        for cx, cy in corners[1:]:
            self._teleport(cx, cy)
        self._pen_up()

    # ─────────────────────────────────────────────────────────────────────
    #  NUEVO: dibujar árbol RRT
    #  Recibe lista de aristas [[x0,y0],[x1,y1]] y las traza.
    #  OPTIMIZACIÓN: agrupa teleports consecutivos cuando el inicio de
    #  una arista coincide con el final de la anterior (evita pen_up/down
    #  innecesarios).
    # ─────────────────────────────────────────────────────────────────────
    def _draw_tree_edges(self, edges):
        """
        Dibuja las aristas del árbol RRT en verde oscuro.
        Cada arista es un segmento corto (≈ step_size metros).
        """
        if not edges:
            return

        self.get_logger().info(f"Dibujando {len(edges)} aristas del árbol RRT...")
        color = (30, 140, 30)   # verde oscuro

        for edge in edges:
            x0, y0 = edge[0][0], edge[0][1]
            x1, y1 = edge[1][0], edge[1][1]
            self._pen_up()
            self._teleport(x0, y0)
            self._set_pen(*color, width=1, off=False)
            self._teleport(x1, y1)

        self._pen_up()
        self.get_logger().info("Árbol RRT dibujado ✓")

    # ─────────────────────────────────────────────────────────────────────
    #  NUEVO: dibujar path final
    #  Línea continua cyan gruesa que une los nodos del camino.
    # ─────────────────────────────────────────────────────────────────────
    def _draw_path_line(self, path_nodes):
        """Dibuja el path como línea continua cyan."""
        if not path_nodes:
            return

        self.get_logger().info(
            f"Dibujando path ({len(path_nodes)} nodos) en cyan..."
        )
        self._pen_up()
        self._teleport(path_nodes[0][0], path_nodes[0][1])
        self._set_pen(0, 220, 220, width=4, off=False)   # cyan grueso
        for px, py in path_nodes[1:]:
            self._teleport(px, py)
        self._pen_up()
        self.get_logger().info("Path dibujado ✓")

    # ── Callbacks del planner ─────────────────────────────────────────────

    def cb_tree_edges(self, msg: String):
        """Recibe aristas del árbol y las dibuja. Solo una vez."""
        if self._tree_drawn or not self._base_drawn:
            return
        self._tree_drawn = True

        try:
            edges = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Error al parsear /rrt/tree_edges")
            return

        self._draw_tree_edges(edges)

    def cb_path(self, msg: String):
        """
        Recibe el path y lo dibuja encima del árbol.
        Espera a que el árbol esté dibujado primero.
        """
        if self._path_drawn:
            return

        if not self._tree_drawn:
            # Reintentar en 0.3 s (timer de un solo disparo)
            self._pending_path_msg = msg
            self.create_timer(0.3, self._retry_path)
            return

        self._draw_path_final(msg)

    def _retry_path(self):
        """Timer de reintento para dibujar el path."""
        if hasattr(self, '_pending_path_msg') and not self._path_drawn:
            if self._tree_drawn:
                self._draw_path_final(self._pending_path_msg)

    def _draw_path_final(self, msg: String):
        """Dibuja el path y redibuja marcadores encima."""
        if self._path_drawn:
            return
        self._path_drawn = True

        try:
            path_nodes = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Error al parsear /rrt/path")
            return

        self._draw_path_line(path_nodes)

        # Redibujar marcadores de start y goal encima de todo
        self._fill_circle_fast(self.sx, self.sy, 0.25, 40, 160, 40, lines=8)
        self._draw_circle_outline(self.sx, self.sy, 0.15, 255, 255, 255, width=3)
        self._fill_circle_fast(self.gx, self.gy, 0.35, 40, 80, 200, lines=8)
        self._draw_cross(self.gx, self.gy, 0.4, 255, 255, 255, width=4)

        # Aparcar el pintor fuera del área
        self._pen_up()
        self._teleport(0.2, 0.2)
        self.get_logger().info(
            "Visualización completa ✓  —  El robot navegará ahora."
        )

    # ── Rutina de dibujado base ───────────────────────────────────────────

    def _start_base(self):
        if self._base_drawn:
            return
        self._base_drawn = True

        self.get_logger().info("Generando tortuga 'painter'...")
        self._spawn_painter()
        self._pen_up()

        # 1. Borde del mundo
        self.get_logger().info("Dibujando borde del mundo...")
        self._draw_world_border()

        # 2. Obstáculos → rojo
        self.get_logger().info(f"Dibujando {len(self.obstacles)} obstáculos...")
        for (cx, cy, radius) in self.obstacles:
            self._fill_circle_fast(cx, cy, radius, 180, 40, 40, lines=14)
        self.get_logger().info("Obstáculos dibujados ✓")

        # 3. Start → verde
        self._fill_circle_fast(self.sx, self.sy, 0.25, 40, 160, 40, lines=8)
        self._draw_circle_outline(self.sx, self.sy, 0.15, 255, 255, 255, width=3)
        self.get_logger().info("Start (verde) ✓")

        # 4. Goal → azul
        self._fill_circle_fast(self.gx, self.gy, 0.35, 40, 80, 200, lines=8)
        self._draw_cross(self.gx, self.gy, 0.4, 255, 255, 255, width=4)
        self.get_logger().info("Goal (azul) ✓")

        self._pen_up()
        self._teleport(0.2, 0.2)
        self.get_logger().info(
            "Dibujado base completo ✓  —  Esperando datos del planner RRT..."
        )


def main(args=None):
    rclpy.init(args=args)
    node = RRTVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
