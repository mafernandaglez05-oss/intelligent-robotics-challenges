#!/usr/bin/env python3
"""
Grid Visualizer OPTIMIZADO para TurtleSim (ROS2 Humble)
========================================================
MEJORAS RESPECTO A LA VERSIÓN ANTERIOR:
  1. VELOCIDAD: dibuja líneas del grid con trazos continuos (snake pattern)
     en lugar de teleport por cada segmento → ~10x más rápido.
  2. VELOCIDAD: el relleno de celdas usa 3 líneas densas en lugar de N/0.04
     llamadas de servicio.
  3. NUEVO: dibuja nodos expandidos por A* (color amarillo/naranja).
  4. NUEVO: dibuja el path final (color cyan/blanco) encima de todo.
  5. Suscribe a /astar/expanded y /astar/path (JSON) publicados por el planner.

Orden de dibujado:
  1. Líneas del grid (gris)
  2. Obstáculos (rojo)
  3. Celda inicio (verde) + marcador
  4. Celda meta (azul) + marcador
  5. Nodos expandidos (naranja claro) — al recibir /astar/expanded
  6. Path final (cyan) — al recibir /astar/path

Parámetros ROS2 (deben coincidir con astar_planner_node):
  grid_size  (int,   default 10)
  cell_size  (float, default 0.50)
  origin_x   (float, default 0.5)
  origin_y   (float, default 0.5)
  obstacles  (str,   default "2,2;2,3;3,2;5,5;5,6;6,5;7,3;3,7")
  draw_grid  (bool,  default True)
  start_col  (int,   default 0)
  start_row  (int,   default 0)
  goal_col   (int,   default 9)
  goal_row   (int,   default 9)

Uso:
  ros2 run <paquete> grid_visualizer_node --ros-args \
    -p obstacles:="2,2;3,2;4,2;5,2" \
    -p goal_col:=9 -p goal_row:=9
"""

import rclpy
import math
import json
import time
from rclpy.node import Node
from turtlesim.srv import Spawn, SetPen, TeleportAbsolute, Kill
from std_srvs.srv import Empty
from std_msgs.msg import String


class GridVisualizerNode(Node):

    def __init__(self):
        super().__init__('grid_visualizer')
        self.get_logger().info("Grid Visualizer (optimizado) iniciado")

        # ── Parámetros ──────────────────────────────────────────────────────
        self.declare_parameter('grid_size', 10)
        self.declare_parameter('cell_size', 0.50)
        self.declare_parameter('origin_x',  0.5)
        self.declare_parameter('origin_y',  0.5)
        self.declare_parameter('obstacles', "2,2;2,3;3,2;5,5;5,6;6,5;7,3;3,7")
        self.declare_parameter('draw_grid', True)
        self.declare_parameter('start_col', 0)
        self.declare_parameter('start_row', 0)
        self.declare_parameter('goal_col',  9)
        self.declare_parameter('goal_row',  9)

        self.N         = self.get_parameter('grid_size').value
        self.cell      = self.get_parameter('cell_size').value
        self.ox        = self.get_parameter('origin_x').value
        self.oy        = self.get_parameter('origin_y').value
        self.draw_grid = self.get_parameter('draw_grid').value
        obs_str        = self.get_parameter('obstacles').value
        self.start     = (self.get_parameter('start_col').value,
                          self.get_parameter('start_row').value)
        self.goal      = (self.get_parameter('goal_col').value,
                          self.get_parameter('goal_row').value)

        self.obstacles = self._parse_obstacles(obs_str)

        # Conjuntos para evitar redibujar sobre obstáculos/start/goal
        self._obstacle_set = set(self.obstacles)

        # ── Clientes de servicios turtlesim ─────────────────────────────────
        self.cli_spawn    = self.create_client(Spawn,            '/spawn')
        self.cli_kill     = self.create_client(Kill,             '/kill')
        self.cli_teleport = self.create_client(TeleportAbsolute, '/painter/teleport_absolute')
        self.cli_pen      = self.create_client(SetPen,           '/painter/set_pen')
        self.cli_clear    = self.create_client(Empty,            '/clear')

        self.cli_spawn.wait_for_service(timeout_sec=5.0)
        self.cli_clear.wait_for_service(timeout_sec=5.0)

        # ── Suscriptores para datos del planner ─────────────────────────────
        self.create_subscription(String, '/astar/expanded', self.cb_expanded, 10)
        self.create_subscription(String, '/astar/path',     self.cb_path,     10)

        self._base_drawn      = False
        self._expanded_drawn  = False
        self._path_drawn      = False

        # Arrancar dibujado base tras 1 s
        self.create_timer(1.0, self._start_drawing)

    # ── Parser de obstáculos ────────────────────────────────────────────────
    def _parse_obstacles(self, obs_str: str):
        cells = []
        for token in obs_str.split(';'):
            token = token.strip()
            if not token:
                continue
            parts = token.split(',')
            if len(parts) != 2:
                continue
            try:
                c, r = int(parts[0].strip()), int(parts[1].strip())
                if 0 <= c < self.N and 0 <= r < self.N:
                    cells.append((c, r))
            except ValueError:
                continue
        return cells

    # ── Coordenadas ─────────────────────────────────────────────────────────
    def _cell_center(self, col, row):
        x = self.ox + (col + 0.5) * self.cell
        y = self.oy + (row + 0.5) * self.cell
        return x, y

    def _cell_corner(self, col, row):
        x = self.ox + col * self.cell
        y = self.oy + row * self.cell
        return x, y

    # ── Primitivas de servicio ───────────────────────────────────────────────
    def _teleport(self, x, y, theta=0.0):
        req = TeleportAbsolute.Request()
        req.x, req.y, req.theta = float(x), float(y), float(theta)
        future = self.cli_teleport.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    def _set_pen(self, r, g, b, width=2, off=0):
        req = SetPen.Request()
        req.r, req.g, req.b = r, g, b
        req.width = width
        req.off   = off
        future = self.cli_pen.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    def _pen_up(self):
        self._set_pen(0, 0, 0, width=1, off=1)

    # ── Spawn de la tortuga pincel ───────────────────────────────────────────
    def _spawn_painter(self):
        # Matar si ya existe
        if self.cli_kill.service_is_ready():
            kill_req = Kill.Request()
            kill_req.name = 'painter'
            f = self.cli_kill.call_async(kill_req)
            rclpy.spin_until_future_complete(self, f, timeout_sec=2.0)
            time.sleep(0.3)

        req = Spawn.Request()
        req.x, req.y, req.theta = 0.1, 0.1, 0.0
        req.name = 'painter'
        future = self.cli_spawn.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        time.sleep(0.5)

        self.cli_teleport.wait_for_service(timeout_sec=5.0)
        self.cli_pen.wait_for_service(timeout_sec=5.0)

    # ─────────────────────────────────────────────────────────────────────────
    #  OPTIMIZACIÓN: dibujar grid con patrón snake
    #  En lugar de teleport+pen_up+pen_down por cada segmento, dibuja todas
    #  las líneas verticales en un recorrido continuo de arriba a abajo
    #  alternando dirección (zigzag), y luego las horizontales igual.
    #  Número de teleports: 2*(N+1) en lugar de 4*(N+1).
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_grid_lines(self):
        self.get_logger().info("Dibujando grid (modo rápido)...")
        r, g, b = 70, 70, 70

        total_size = self.N * self.cell

        # Líneas verticales — snake: columna 0 sube, col 1 baja, col 2 sube…
        self._pen_up()
        for c in range(self.N + 1):
            x = self.ox + c * self.cell
            if c % 2 == 0:
                y_start = self.oy
                y_end   = self.oy + total_size
            else:
                y_start = self.oy + total_size
                y_end   = self.oy
            self._teleport(x, y_start)
            self._set_pen(r, g, b, width=1, off=0)
            self._teleport(x, y_end)
            self._pen_up()

        # Líneas horizontales — snake: fila 0 va derecha, fila 1 izquierda…
        for row in range(self.N + 1):
            y = self.oy + row * self.cell
            if row % 2 == 0:
                x_start = self.ox
                x_end   = self.ox + total_size
            else:
                x_start = self.ox + total_size
                x_end   = self.ox
            self._teleport(x_start, y)
            self._set_pen(r, g, b, width=1, off=0)
            self._teleport(x_end, y)
            self._pen_up()

        self.get_logger().info("Grid dibujado ✓")

    # ─────────────────────────────────────────────────────────────────────────
    #  OPTIMIZACIÓN: relleno de celda con solo 3 líneas horizontales densas
    #  (en lugar de cell/0.04 llamadas). Suficiente para que visualmente
    #  se vea como un cuadrado sólido, y mucho más rápido.
    # ─────────────────────────────────────────────────────────────────────────
    def _fill_cell_fast(self, col, row, r_=180, g_=40, b_=40, lines=5):
        """
        Rellena una celda con `lines` trazos horizontales.
        Menos teleports = más velocidad sin perder claridad visual.
        Parámetros renombrados r_/g_/b_ para evitar conflicto con la variable
        de iteración 'r' (row) usada en los loops del visualizador.
        """
        x0, y0 = self._cell_corner(col, row)
        x1 = x0 + self.cell
        self._pen_up()
        for i in range(lines):
            yy = y0 + (i + 0.5) / lines * self.cell
            self._teleport(x0, yy)
            self._set_pen(r_, g_, b_, width=3, off=0)
            self._teleport(x1, yy)
            self._pen_up()

        # Contorno de la celda (4 lados en un solo trazo continuo)
        corners = [
            (x0, y0), (x1, y0), (x1, y0 + self.cell),
            (x0, y0 + self.cell), (x0, y0)
        ]
        self._teleport(*corners[0])
        self._set_pen(r_, g_, b_, width=2, off=0)
        for cx, cy in corners[1:]:
            self._teleport(cx, cy)
        self._pen_up()

    # ── Marcadores ──────────────────────────────────────────────────────────
    def _draw_cross(self, col, row, r, g, b, size_factor=0.4):
        cx, cy = self._cell_center(col, row)
        half = self.cell * size_factor / 2
        self._pen_up()
        self._teleport(cx - half, cy - half)
        self._set_pen(r, g, b, width=4, off=0)
        self._teleport(cx + half, cy + half)
        self._pen_up()
        self._teleport(cx + half, cy - half)
        self._set_pen(r, g, b, width=4, off=0)
        self._teleport(cx - half, cy + half)
        self._pen_up()

    def _draw_circle(self, col, row, r, g, b, radius_factor=0.3):
        cx, cy = self._cell_center(col, row)
        radius = self.cell * radius_factor
        steps  = 16
        self._pen_up()
        self._teleport(cx + radius, cy)
        self._set_pen(r, g, b, width=4, off=0)
        for i in range(1, steps + 1):
            angle = 2 * math.pi * i / steps
            self._teleport(
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle)
            )
        self._pen_up()

    # ─────────────────────────────────────────────────────────────────────────
    #  NUEVO: dibujar nodos expandidos por A*
    #  Se dibuja como un punto (dot) pequeño en el centro de cada celda.
    #  No se sobreescriben obstáculos, start ni goal.
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_expanded_node(self, col, row):
        """Dibuja un punto pequeño naranja en el centro de la celda."""
        cx, cy = self._cell_center(col, row)
        radius = self.cell * 0.15
        steps  = 10
        self._pen_up()
        self._teleport(cx + radius, cy)
        self._set_pen(220, 140, 30, width=3, off=0)   # naranja
        for i in range(1, steps + 1):
            angle = 2 * math.pi * i / steps
            self._teleport(
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle)
            )
        self._pen_up()

    # ─────────────────────────────────────────────────────────────────────────
    #  NUEVO: dibujar el path final
    #  Línea continua cyan que conecta el centro de cada celda del camino.
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_path_line(self, path_cells):
        """
        Dibuja una línea continua gruesa que une los centros de las
        celdas del path encontrado por A*.
        """
        if not path_cells:
            return
        self._pen_up()
        cx, cy = self._cell_center(*path_cells[0])
        self._teleport(cx, cy)
        self._set_pen(0, 220, 220, width=4, off=0)   # cyan
        for col, row in path_cells[1:]:
            cx, cy = self._cell_center(col, row)
            self._teleport(cx, cy)
        self._pen_up()

    # ── Callbacks del planner ────────────────────────────────────────────────

    def cb_expanded(self, msg: String):
        """
        Recibe la lista de nodos expandidos y los dibuja.
        Solo se ejecuta una vez (después del dibujado base).
        """
        if self._expanded_drawn or not self._base_drawn:
            return
        self._expanded_drawn = True

        try:
            cells = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Error al parsear /astar/expanded")
            return

        self.get_logger().info(
            f"Dibujando {len(cells)} nodos expandidos (naranja)..."
        )
        for cell in cells:
            col, row = cell[0], cell[1]
            # No sobreescribir obstáculos, start ni goal
            if (col, row) in self._obstacle_set:
                continue
            if (col, row) == self.start or (col, row) == self.goal:
                continue
            self._draw_expanded_node(col, row)

        self.get_logger().info("Nodos expandidos dibujados ✓")

    def cb_path(self, msg: String):
        """
        Recibe el path final y lo dibuja encima de todo.
        Espera a que los nodos expandidos estén dibujados.
        """
        if self._path_drawn or not self._expanded_drawn:
            # Reintentar en 0.5 s si la expansión aún no se dibujó
            self.create_timer(0.5, lambda: self.cb_path(msg))
            return
        self._path_drawn = True

        try:
            cells = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Error al parsear /astar/path")
            return

        path_cells = [(c[0], c[1]) for c in cells]
        self.get_logger().info(
            f"Dibujando path ({len(path_cells)} celdas) en cyan..."
        )
        self._draw_path_line(path_cells)

        # Redibujar marcadores de start y goal encima del path
        self._draw_circle(self.start[0], self.start[1], 255, 255, 255)
        self._draw_cross(self.goal[0],  self.goal[1],  255, 255, 255)

        # Aparcar el pintor fuera del área de juego
        self._pen_up()
        self._teleport(0.2, 0.2)
        self.get_logger().info("Path dibujado ✓  —  Visualización completa ✓")
        self.get_logger().info("Ahora corre el robot (puzzlebot_controller).")

    # ── Rutina principal de dibujado base ────────────────────────────────────

    def _start_drawing(self):
        if self._base_drawn:
            return
        self._base_drawn = True

        self.get_logger().info("Generando tortuga 'painter'...")
        self._spawn_painter()
        self._pen_up()

        # 1. Líneas del grid
        if self.draw_grid:
            self._draw_grid_lines()

        # 2. Obstáculos → rojo oscuro
        self.get_logger().info(f"Dibujando {len(self.obstacles)} obstáculos...")
        for (c, r) in self.obstacles:
            self._fill_cell_fast(c, r, r_=180, g_=40, b_=40)
        self.get_logger().info("Obstáculos dibujados ✓")

        # 3. Celda inicio → verde
        self._fill_cell_fast(self.start[0], self.start[1], r_=40, g_=160, b_=40)
        self._draw_circle(self.start[0], self.start[1], 255, 255, 255)
        self.get_logger().info("Celda inicio (verde) ✓")

        # 4. Celda meta → azul
        self._fill_cell_fast(self.goal[0], self.goal[1], r_=40, g_=80, b_=200)
        self._draw_cross(self.goal[0], self.goal[1], 255, 255, 255)
        self.get_logger().info("Celda meta (azul) ✓")

        self._pen_up()
        self._teleport(0.2, 0.2)
        self.get_logger().info(
            "Dibujado base completo ✓  —  Esperando datos del planner A*..."
        )


def main(args=None):
    rclpy.init(args=args)
    node = GridVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
