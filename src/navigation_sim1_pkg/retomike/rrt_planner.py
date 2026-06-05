#!/usr/bin/env python3
"""
Reto 2 — RRT Planner para TurtleSim (ROS2 Humble)
===================================================
CORRECCIONES respecto a la versión anterior:
  1. BUG FIX (race condition): en run_logic, self.waiting = True se asigna
     ANTES de pub_goal.publish(). En TurtleSim el controller puede responder
     tan rápido que cb_goal_reached llegaba antes de que waiting fuera True,
     haciendo que se saltaran waypoints silenciosamente.
  2. BUG FIX: goal.parent se resetea a None al inicio de build_rrt para
     evitar que un objeto goal reutilizado tenga un parent "contaminado"
     de una llamada anterior, lo que generaría un path incorrecto.

Arquitectura:
  Construye un árbol RRT en el espacio continuo de TurtleSim, extrae el
  camino start→goal y publica los waypoints uno a uno al controller.
  Publica las aristas del árbol y el path completo como JSON
  para que el rrt_visualizer los dibuje en TurtleSim.

Tópicos:
  Publica  → /goals            (turtlesim/msg/Pose)      waypoint actual
  Publica  → /rrt/tree_edges   (std_msgs/msg/String)     aristas del árbol (JSON)
  Publica  → /rrt/path         (std_msgs/msg/String)     path final (JSON)
  Suscribe → /goal_reached     (std_msgs/msg/Bool)       confirmación del controller
  Suscribe → /odom             (geometry_msgs/msg/Pose2D) posición actual

Parámetros ROS2:
  start_x     (float, default 1.0)
  start_y     (float, default 1.0)
  goal_x      (float, default 9.0)
  goal_y      (float, default 9.0)
  goal_radius (float, default 0.4)
  max_iter    (int,   default 2000)
  step_size   (float, default 0.5)   — ε: tamaño de paso al extender
  world_min   (float, default 0.5)
  world_max   (float, default 10.5)
  goal_bias   (float, default 0.15)  — prob. de muestrear el goal directo
  obstacles   (str, default "3.0,3.0,0.8;6.0,4.0,0.8;4.0,7.0,0.8;7.0,7.5,0.8")
              formato: "cx,cy,radio;..."  — obstáculos circulares
"""

import rclpy
import math
import random
import json
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Bool, String


# ─────────────────────────────────────────────────────────────────────────────
#  Estructuras de datos
# ─────────────────────────────────────────────────────────────────────────────

class Node2D:
    """Nodo del árbol RRT."""
    __slots__ = ('x', 'y', 'parent')

    def __init__(self, x: float, y: float):
        self.x      = x
        self.y      = y
        self.parent = None   # Node2D padre (None para la raíz)

    def __repr__(self):
        return f"Node2D({self.x:.2f}, {self.y:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
#  Algoritmo RRT  (espacio continuo 2D)
# ─────────────────────────────────────────────────────────────────────────────

def dist(a: Node2D, b: Node2D) -> float:
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def steer(from_node: Node2D, to_node: Node2D, step: float) -> Node2D:
    """
    Extiende desde from_node hacia to_node un máximo de `step` metros.

    Ecuación:
        q_new = q_near + ε · (q_rand − q_near) / ‖q_rand − q_near‖

    Si la distancia ya es menor que ε, devuelve q_rand directamente
    (el árbol "alcanza" el punto muestreado sin sobrepasarlo).
    """
    d = dist(from_node, to_node)
    if d < 1e-9:
        return Node2D(from_node.x, from_node.y)
    ratio = min(step / d, 1.0)
    nx = from_node.x + ratio * (to_node.x - from_node.x)
    ny = from_node.y + ratio * (to_node.y - from_node.y)
    return Node2D(nx, ny)


def segment_circle_collision(a: Node2D, b: Node2D,
                              cx: float, cy: float, r: float) -> bool:
    """
    Verifica si el segmento AB colisiona con el círculo (cx, cy, r).

    Método: proyecta el centro del círculo sobre la recta AB, clampea
    el parámetro t ∈ [0,1] para quedarse en el segmento y comprueba
    si la distancia al punto más cercano es menor que r.
    """
    dx = b.x - a.x
    dy = b.y - a.y
    fx = a.x - cx
    fy = a.y - cy

    length_sq = dx*dx + dy*dy
    if length_sq < 1e-12:
        return math.sqrt(fx*fx + fy*fy) < r

    t = -(fx*dx + fy*dy) / length_sq
    t = max(0.0, min(1.0, t))

    closest_x = a.x + t*dx - cx
    closest_y = a.y + t*dy - cy
    return (closest_x**2 + closest_y**2) < r * r


def is_collision_free(a: Node2D, b: Node2D, obstacles: list) -> bool:
    """True si el segmento AB no colisiona con ningún obstáculo."""
    for (cx, cy, r) in obstacles:
        if segment_circle_collision(a, b, cx, cy, r):
            return False
    return True


def point_in_obstacle(x: float, y: float, obstacles: list) -> bool:
    """True si el punto (x, y) cae dentro de algún obstáculo."""
    for (cx, cy, r) in obstacles:
        if math.sqrt((x - cx)**2 + (y - cy)**2) < r:
            return True
    return False


def build_rrt(start: Node2D, goal: Node2D,
              obstacles: list,
              world_min: float, world_max: float,
              step_size: float, max_iter: int,
              goal_radius: float, goal_bias: float):
    """
    Construye el árbol RRT desde start hacia goal.

    CORRECCIÓN: goal.parent se resetea a None al inicio para evitar
    que un objeto reutilizado tenga un parent "contaminado" que genere
    un path incorrecto al reconstruir con los punteros parent.

    Algoritmo completo:
      1. Inicializa el árbol T = {q_start}.
      2. Para cada iteración i = 1..max_iter:
           a. q_rand ← sample_free()     (con prob. goal_bias → q_goal)
           b. q_near ← nearest(T, q_rand)
           c. q_new  ← steer(q_near, q_rand, ε)
           d. Si el segmento q_near→q_new es libre: agregar q_new a T.
           e. Si dist(q_new, q_goal) ≤ goal_radius y el segmento
              q_new→q_goal es libre: conectar goal y terminar.
      3. Reconstruir camino siguiendo punteros parent desde goal.

    Propiedad clave — Completitud probabilística:
      Con suficientes iteraciones, la probabilidad de encontrar un camino
      (si existe) tiende a 1. No garantiza encontrarlo en un número fijo
      de iteraciones (a diferencia de A* que es completo en grids finitos).

    Returns:
        (tree, path, edges)
        tree  : lista de todos los Node2D en el árbol
        path  : lista de Node2D desde start hasta goal, o [] si no encontró
        edges : lista de ((x0,y0),(x1,y1)) de cada arista del árbol
    """
    # FIX: reset goal.parent para evitar paths incorrectos por reutilización
    goal.parent = None

    tree  = [start]
    edges = []

    for _ in range(max_iter):

        # ── a. Muestreo ───────────────────────────────────────────────────
        if random.random() < goal_bias:
            q_rand = Node2D(goal.x, goal.y)
        else:
            rx = random.uniform(world_min, world_max)
            ry = random.uniform(world_min, world_max)
            q_rand = Node2D(rx, ry)

        # ── b. Nodo más cercano ───────────────────────────────────────────
        q_near = min(tree, key=lambda n: dist(n, q_rand))

        # ── c. Extensión ──────────────────────────────────────────────────
        q_new = steer(q_near, q_rand, step_size)

        # ── d. Verificación de colisión ───────────────────────────────────
        if point_in_obstacle(q_new.x, q_new.y, obstacles):
            continue
        if not is_collision_free(q_near, q_new, obstacles):
            continue

        # ── Agregar nodo y arista al árbol ────────────────────────────────
        q_new.parent = q_near
        tree.append(q_new)
        edges.append(((q_near.x, q_near.y), (q_new.x, q_new.y)))

        # ── e. ¿Llegamos al goal? ─────────────────────────────────────────
        if dist(q_new, goal) <= goal_radius:
            if is_collision_free(q_new, goal, obstacles):
                goal.parent = q_new
                tree.append(goal)
                edges.append(((q_new.x, q_new.y), (goal.x, goal.y)))

                # Reconstruir camino
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = node.parent
                path.reverse()
                return tree, path, edges

    return tree, [], edges   # No encontró camino


# ─────────────────────────────────────────────────────────────────────────────
#  Nodo ROS2
# ─────────────────────────────────────────────────────────────────────────────

class RRTPlannerNode(Node):

    def __init__(self):
        super().__init__('rrt_planner')
        self.get_logger().info("RRT Planner iniciado")

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('start_x',     1.0)
        self.declare_parameter('start_y',     1.0)
        self.declare_parameter('goal_x',      10.0)
        self.declare_parameter('goal_y',      10.0)
        self.declare_parameter('goal_radius', 0.4)
        self.declare_parameter('max_iter',    5000)
        self.declare_parameter('step_size',   0.5)
        self.declare_parameter('world_min',   0.5)
        self.declare_parameter('world_max',   10.5)
        self.declare_parameter('goal_bias',   0.5)
        self.declare_parameter('obstacles',
            "2.0,8.0,0.9;4.5,8.0,0.9;7.5,8.0,0.9;2.0,5.0,0.9;4.5,5.0,0.9;7.5,5.0,0.9")

        sx        = self.get_parameter('start_x').value
        sy        = self.get_parameter('start_y').value
        gx        = self.get_parameter('goal_x').value
        gy        = self.get_parameter('goal_y').value
        goal_r    = self.get_parameter('goal_radius').value
        max_iter  = self.get_parameter('max_iter').value
        step_size = self.get_parameter('step_size').value
        world_min = self.get_parameter('world_min').value
        world_max = self.get_parameter('world_max').value
        goal_bias = self.get_parameter('goal_bias').value
        obs_str   = self.get_parameter('obstacles').value

        self.obstacles = self._parse_obstacles(obs_str)

        # ── Ejecutar RRT ──────────────────────────────────────────────────
        random.seed()
        start = Node2D(sx, sy)
        goal  = Node2D(gx, gy)

        self.get_logger().info(
            f"Construyendo árbol RRT: start=({sx},{sy}) goal=({gx},{gy}) "
            f"max_iter={max_iter} step={step_size} goal_bias={goal_bias}"
        )

        self.tree, path, self.edges = build_rrt(
            start, goal, self.obstacles,
            world_min, world_max,
            step_size, max_iter, goal_r, goal_bias
        )

        if not path:
            self.get_logger().error(
                f"RRT no encontró camino en {max_iter} iteraciones. "
                "Prueba aumentar max_iter o goal_bias."
            )
            self.waypoints  = []
            self.path_nodes = []
        else:
            self.waypoints  = [(n.x, n.y) for n in path]
            self.path_nodes = [(n.x, n.y) for n in path]
            path_len = sum(
                math.sqrt(
                    (path[i+1].x - path[i].x)**2 +
                    (path[i+1].y - path[i].y)**2
                )
                for i in range(len(path) - 1)
            )
            self.get_logger().info(
                f"Camino encontrado: {len(self.waypoints)} waypoints | "
                f"longitud={path_len:.2f} m | "
                f"nodos en árbol={len(self.tree)} | "
                f"aristas={len(self.edges)}"
            )
            self._log_tree_ascii(self.tree, path, world_min, world_max)

        # ── Estado de navegación ──────────────────────────────────────────
        self.current_wp   = 0
        # FIX: waiting arranca en False, se pone True ANTES de publicar
        self.waiting      = False
        self._done_logged = False

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_goal  = self.create_publisher(Pose,   '/goals',          10)
        self.pub_edges = self.create_publisher(String, '/rrt/tree_edges',  10)
        self.pub_path  = self.create_publisher(String, '/rrt/path',        10)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Bool,   '/goal_reached', self.cb_goal_reached, 10)
        self.create_subscription(Pose2D, '/odom',         self.cb_odom,          1)

        # ── Timers ────────────────────────────────────────────────────────
        self._viz_published = False
        self.create_timer(2.0, self._publish_viz_data)
        self.create_timer(0.05, self.run_logic)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _parse_obstacles(self, obs_str: str):
        obstacles = []
        if not obs_str.strip():
            return obstacles
        for token in obs_str.split(';'):
            token = token.strip()
            if not token:
                continue
            parts = token.split(',')
            if len(parts) != 3:
                self.get_logger().warn(f"Obstáculo mal formado: '{token}'")
                continue
            try:
                cx, cy, r = float(parts[0]), float(parts[1]), float(parts[2])
                obstacles.append((cx, cy, r))
            except ValueError:
                self.get_logger().warn(f"Obstáculo no numérico: '{token}'")
        return obstacles

    def _publish_viz_data(self):
        """
        Publica una sola vez las aristas del árbol y el path como JSON
        para que el rrt_visualizer los dibuje en TurtleSim.

        Formato aristas: [[[x0,y0],[x1,y1]], ...]
        Formato path:    [[x0,y0], [x1,y1], ...]
        """
        if self._viz_published:
            return
        self._viz_published = True

        edges_msg      = String()
        edges_msg.data = json.dumps(
            [[[e[0][0], e[0][1]], [e[1][0], e[1][1]]] for e in self.edges]
        )
        self.pub_edges.publish(edges_msg)
        self.get_logger().info(
            f"Publicadas {len(self.edges)} aristas → /rrt/tree_edges"
        )

        path_msg      = String()
        path_msg.data = json.dumps([[p[0], p[1]] for p in self.path_nodes])
        self.pub_path.publish(path_msg)
        self.get_logger().info(
            f"Publicado path ({len(self.path_nodes)} nodos) → /rrt/path"
        )

    def _log_tree_ascii(self, tree, path, wmin, wmax, res=20):
        """Imprime el árbol y camino en una cuadrícula ASCII 20x20."""
        grid     = [['.' for _ in range(res)] for _ in range(res)]

        def to_cell(x, y):
            c = int((x - wmin) / (wmax - wmin) * (res - 1))
            r = int((y - wmin) / (wmax - wmin) * (res - 1))
            return max(0, min(res-1, c)), max(0, min(res-1, r))

        for (cx, cy, r) in self.obstacles:
            for row in range(res):
                for col in range(res):
                    wx = wmin + col / (res-1) * (wmax - wmin)
                    wy = wmin + row / (res-1) * (wmax - wmin)
                    if math.sqrt((wx-cx)**2 + (wy-cy)**2) < r:
                        grid[row][col] = 'O'

        for n in tree:
            c, r = to_cell(n.x, n.y)
            if grid[r][c] == '.':
                grid[r][c] = '+'

        for n in path:
            c, r = to_cell(n.x, n.y)
            grid[r][c] = '*'

        if path:
            sc, sr = to_cell(path[0].x,  path[0].y)
            gc, gr = to_cell(path[-1].x, path[-1].y)
            grid[sr][sc] = 'S'
            grid[gr][gc] = 'G'

        lines = [
            f"\nÁrbol RRT  "
            f"(+ nodo árbol | * camino | O obstáculo | S start | G goal)"
        ]
        for r in range(res - 1, -1, -1):
            lines.append("  " + " ".join(grid[r]))
        self.get_logger().info("\n".join(lines))

    # ── Callbacks ─────────────────────────────────────────────────────────

    def cb_goal_reached(self, msg: Bool):
        if msg.data and self.waiting:
            self.get_logger().info(
                f"Waypoint {self.current_wp}/{len(self.waypoints)-1} alcanzado"
            )
            self.current_wp += 1
            self.waiting = False

    def cb_odom(self, msg: Pose2D):
        pass

    # ── Lógica principal ──────────────────────────────────────────────────

    def run_logic(self):
        if not self.waypoints:
            return

        if self.current_wp >= len(self.waypoints):
            if not self._done_logged:
                self.get_logger().info("¡Destino final alcanzado! RRT completado ✓")
                self._done_logged = True
            return

        if self.waiting:
            return

        x, y = self.waypoints[self.current_wp]
        msg  = Pose()
        msg.x = float(x)
        msg.y = float(y)

        # FIX: waiting = True ANTES de publish para evitar el race condition
        # donde cb_goal_reached llega antes de que waiting sea True y el
        # waypoint se salta silenciosamente.
        self.waiting = True
        self.pub_goal.publish(msg)

        self.get_logger().info(
            f"→ Waypoint {self.current_wp + 1}/{len(self.waypoints)}: "
            f"({x:.2f}, {y:.2f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = RRTPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
