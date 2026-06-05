#!/usr/bin/env python3
"""
Reto 2 — A* Grid Planner para TurtleSim (ROS2 Humble)
======================================================
Arquitectura:
  Este nodo calcula un camino libre de obstáculos usando A* sobre un grid
  y publica los waypoints uno a uno al puzzlebot_controller existente.
  NUEVO: publica nodos expandidos y el path completo para visualización.

Tópicos:
  Publica  → /goals              (turtlesim/msg/Pose)        waypoint actual
  Publica  → /astar/expanded     (std_msgs/msg/String)       nodos expandidos (JSON)
  Publica  → /astar/path         (std_msgs/msg/String)       path final (JSON)
  Suscribe → /goal_reached       (std_msgs/msg/Bool)         confirmación del controlador
  Suscribe → /odom               (geometry_msgs/msg/Pose2D)  posición actual

Parámetros ROS2:
  grid_size      (int,   default 10)
  cell_size      (float, default 0.50)
  origin_x       (float, default 0.5)
  origin_y       (float, default 0.5)
  start_col      (int,   default 0)
  start_row      (int,   default 0)
  goal_col       (int,   default 9)
  goal_row       (int,   default 9)
  obstacles      (str,   default "2,2;2,3;3,2;5,5;5,6;6,5;7,3;3,7")
"""

import rclpy
import math
import heapq
import json
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Bool, String


# ─────────────────────────────────────────────────────────────────────────────
#  Algoritmo A*  (puro Python, sin dependencias externas)
# ─────────────────────────────────────────────────────────────────────────────

def heuristic(a, b):
    """
    Distancia Euclidiana como heurística admisible.
    Es admisible porque nunca sobreestima el costo real al usar
    conectividad-8 con costos 1.0 (cardinal) y √2 (diagonal).
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def astar(grid, start, goal):
    """
    Busca el camino de menor costo en un grid 2D con conectividad 8.

    Args:
        grid  : lista de listas, 0=libre, 1=obstáculo.  grid[row][col]
        start : (col, row) celda inicial
        goal  : (col, row) celda objetivo

    Returns:
        (path, expanded_order)
          path           : lista de (col,row) desde start hasta goal inclusive,
                           o None si no existe camino.
          expanded_order : lista de (col,row) en el orden en que fueron
                           expandidos (extraídos del heap), incluyendo start.
    """
    rows = len(grid)
    cols = len(grid[0])

    # Costos de movimiento: cardinal = 1.0, diagonal = √2
    neighbors_8 = [
        ( 1,  0, 1.0),
        (-1,  0, 1.0),
        ( 0,  1, 1.0),
        ( 0, -1, 1.0),
        ( 1,  1, math.sqrt(2)),
        (-1,  1, math.sqrt(2)),
        ( 1, -1, math.sqrt(2)),
        (-1, -1, math.sqrt(2)),
    ]

    open_heap    = []          # min-heap: (f_cost, col, row)
    past_cost    = {}          # (col,row) → g-cost mínimo encontrado
    parent       = {}          # (col,row) → nodo anterior en el camino
    expanded_order = []        # NUEVO: orden de expansión de nodos

    past_cost[start] = 0.0
    est = heuristic(start, goal)
    heapq.heappush(open_heap, (est, start))

    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        expanded_order.append(current)   # NUEVO: registrar expansión

        if current == goal:
            # Reconstruir camino hacia atrás desde goal hasta start
            path = []
            node = goal
            while node in parent:
                path.append(node)
                node = parent[node]
            path.append(start)
            path.reverse()
            return path, expanded_order

        c, r = current
        for dc, dr, move_cost in neighbors_8:
            nc, nr = c + dc, r + dr
            nbr = (nc, nr)

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == 1:
                continue
            if nbr in closed:
                continue

            # Evitar "cortar esquinas" en movimiento diagonal:
            # si alguno de los dos vecinos cardinales del diagonal es obstáculo,
            # no se permite ese movimiento diagonal.
            if dc != 0 and dr != 0:
                if grid[r][nc] == 1 or grid[nr][c] == 1:
                    continue

            tentative_g = past_cost[current] + move_cost
            if tentative_g < past_cost.get(nbr, float('inf')):
                past_cost[nbr] = tentative_g
                parent[nbr]    = current
                f_cost         = tentative_g + heuristic(nbr, goal)
                heapq.heappush(open_heap, (f_cost, nbr))

    return None, expanded_order   # Sin solución


# ─────────────────────────────────────────────────────────────────────────────
#  Nodo ROS2
# ─────────────────────────────────────────────────────────────────────────────

class AStarPlannerNode(Node):

    def __init__(self):
        super().__init__('astar_planner')
        self.get_logger().info("A* Planner iniciado")

        # ── Parámetros ──────────────────────────────────────────────────────
        self.declare_parameter('grid_size', 10)
        self.declare_parameter('cell_size', 0.50)
        self.declare_parameter('origin_x',  0.5)
        self.declare_parameter('origin_y',  0.5)
        self.declare_parameter('start_col', 0)
        self.declare_parameter('start_row', 0)
        self.declare_parameter('goal_col',  9)
        self.declare_parameter('goal_row',  9)
        self.declare_parameter('obstacles', "2,2;2,3;3,2;5,5;5,6;6,5;7,3;3,7")

        self.grid_size = self.get_parameter('grid_size').value
        self.cell_size = self.get_parameter('cell_size').value
        self.origin_x  = self.get_parameter('origin_x').value
        self.origin_y  = self.get_parameter('origin_y').value
        start_col      = self.get_parameter('start_col').value
        start_row      = self.get_parameter('start_row').value
        goal_col       = self.get_parameter('goal_col').value
        goal_row       = self.get_parameter('goal_row').value
        obstacles_str  = self.get_parameter('obstacles').value

        # ── Construir grid ──────────────────────────────────────────────────
        N = self.grid_size
        self.grid = [[0] * N for _ in range(N)]
        self.obstacles_cells = self._parse_obstacles(obstacles_str, N)
        for (c, r) in self.obstacles_cells:
            self.grid[r][c] = 1

        # ── Ejecutar A* ─────────────────────────────────────────────────────
        start = (start_col, start_row)
        goal  = (goal_col,  goal_row)
        self._log_grid(start, goal)

        path, expanded = astar(self.grid, start, goal)

        if path is None:
            self.get_logger().error(
                f"A* no encontró camino de {start} a {goal}. "
                "Revisa los obstáculos o el grid."
            )
            self.waypoints       = []
            self.path_cells      = []
            self.expanded_cells  = expanded
        else:
            self.waypoints      = [self._cell_to_world(c, r) for (c, r) in path]
            self.path_cells     = path
            self.expanded_cells = expanded
            self.get_logger().info(
                f"Camino encontrado: {len(self.waypoints)} waypoints, "
                f"{len(expanded)} nodos expandidos"
            )
            self._log_path(path)

        # ── Estado de navegación ────────────────────────────────────────────
        self.current_wp  = 0
        self.waiting     = False
        self._done_logged = False

        # ── Publishers ──────────────────────────────────────────────────────
        self.pub_goal     = self.create_publisher(Pose,   '/goals',          10)
        self.pub_expanded = self.create_publisher(String, '/astar/expanded',  10)
        self.pub_path     = self.create_publisher(String, '/astar/path',      10)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.create_subscription(Bool,   '/goal_reached', self.cb_goal_reached, 10)
        self.create_subscription(Pose2D, '/odom',         self.cb_odom,          1)

        # ── Timers ──────────────────────────────────────────────────────────
        # Publica expansión y path al visualizador (arrancan tras 1.5 s para
        # que el visualizador ya esté suscrito antes de recibir los datos)
        self.create_timer(1.5, self._publish_viz_data)
        self._viz_published = False

        self.create_timer(0.5, self.run_logic)

    # ── Helpers de conversión ───────────────────────────────────────────────

    def _parse_obstacles(self, obs_str: str, N: int):
        cells = []
        if not obs_str.strip():
            return cells
        for token in obs_str.split(';'):
            token = token.strip()
            if not token:
                continue
            parts = token.split(',')
            if len(parts) != 2:
                self.get_logger().warn(f"Obstáculo mal formado ignorado: '{token}'")
                continue
            try:
                c, r = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                self.get_logger().warn(f"Obstáculo no numérico ignorado: '{token}'")
                continue
            if 0 <= c < N and 0 <= r < N:
                cells.append((c, r))
            else:
                self.get_logger().warn(
                    f"Obstáculo ({c},{r}) fuera del grid {N}x{N}, ignorado"
                )
        return cells

    def _cell_to_world(self, col: int, row: int):
        """Centro de la celda (col, row) en metros (coordenadas TurtleSim)."""
        x = self.origin_x + (col + 0.5) * self.cell_size
        y = self.origin_y + (row + 0.5) * self.cell_size
        return (x, y)

    # ── Publicar datos de visualización ────────────────────────────────────

    def _publish_viz_data(self):
        """
        Publica una sola vez la lista de nodos expandidos y el path
        como JSON para que el grid_visualizer los dibuje.
        """
        if self._viz_published:
            return
        self._viz_published = True

        # Nodos expandidos: lista de [col, row]
        expanded_msg = String()
        expanded_msg.data = json.dumps([[c, r] for (c, r) in self.expanded_cells])
        self.pub_expanded.publish(expanded_msg)
        self.get_logger().info(
            f"Publicados {len(self.expanded_cells)} nodos expandidos → /astar/expanded"
        )

        # Path final: lista de [col, row]
        path_msg = String()
        path_msg.data = json.dumps([[c, r] for (c, r) in self.path_cells])
        self.pub_path.publish(path_msg)
        self.get_logger().info(
            f"Publicado path ({len(self.path_cells)} celdas) → /astar/path"
        )

    # ── Logging visual del grid en consola ────────────────────────────────

    def _log_grid(self, start, goal):
        N = self.grid_size
        lines = [f"\nGrid {N}x{N}   S=start  G=goal  X=obstáculo  .=libre"]
        for r in range(N - 1, -1, -1):
            row_str = ""
            for c in range(N):
                if (c, r) == start:
                    row_str += " S"
                elif (c, r) == goal:
                    row_str += " G"
                elif self.grid[r][c] == 1:
                    row_str += " X"
                else:
                    row_str += " ."
            lines.append(f"  r{r:02d} |{row_str} |")
        lines.append("       " + "".join(f" {c}" for c in range(N)))
        self.get_logger().info("\n".join(lines))

    def _log_path(self, path):
        N = self.grid_size
        path_set = set(path)
        start    = path[0]
        goal     = path[-1]
        lines    = ["\nCamino A* encontrado  (* = camino):"]
        for r in range(N - 1, -1, -1):
            row_str = ""
            for c in range(N):
                if (c, r) == start:
                    row_str += " S"
                elif (c, r) == goal:
                    row_str += " G"
                elif self.grid[r][c] == 1:
                    row_str += " X"
                elif (c, r) in path_set:
                    row_str += " *"
                else:
                    row_str += " ."
            lines.append(f"  r{r:02d} |{row_str} |")
        lines.append("       " + "".join(f" {c}" for c in range(N)))
        cost = sum(
            math.sqrt(
                (path[i+1][0] - path[i][0])**2 +
                (path[i+1][1] - path[i][1])**2
            )
            for i in range(len(path) - 1)
        )
        lines.append(f"\n  Costo total del camino: {cost:.3f} celdas")
        self.get_logger().info("\n".join(lines))

    # ── Callbacks ROS ─────────────────────────────────────────────────────

    def cb_goal_reached(self, msg: Bool):
        if msg.data and self.waiting:
            self.get_logger().info(
                f"Waypoint {self.current_wp}/{len(self.waypoints)-1} alcanzado"
            )
            self.current_wp += 1
            self.waiting = False

    def cb_odom(self, msg: Pose2D):
        pass  # disponible para debug o visualización futura

    # ── Lógica principal de navegación ────────────────────────────────────

    def run_logic(self):
        if not self.waypoints:
            return

        if self.current_wp >= len(self.waypoints):
            if not self._done_logged:
                self.get_logger().info("¡Destino final alcanzado! A* completado ✓")
                self._done_logged = True
            return

        if self.waiting:
            return

        x, y = self.waypoints[self.current_wp]
        msg  = Pose()
        msg.x = float(x)
        msg.y = float(y)
        self.pub_goal.publish(msg)
        self.waiting = True
        self.get_logger().info(
            f"→ Waypoint {self.current_wp + 1}/{len(self.waypoints)}: "
            f"({x:.2f}, {y:.2f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
