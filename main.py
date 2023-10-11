import graphlib as gl

import pyglet
from pyglet import shapes


def prepare_random_graph(g_size, width, height):
    #graph.print_weitghts()
    return graph


def prim_algorithm(graph):
    # Inicializa el árbol de expansión mínima y la lista de vértices visitados
    mst = []
    visited = [False] * len(graph.nodes)

    # Elije un nodo inicial (puedes elegir cualquier nodo)
    start_node = graph.nodes[0]

    # Marca el nodo inicial como visitado
    visited[graph.nodes.index(start_node)] = True

    # Itera hasta que todos los nodos estén en el árbol de expansión mínima
    while len(mst) < len(graph.nodes) - 1:
        min_weight = float('inf')
        min_edge = None

        # Busca el borde más corto que conecta un vértice visitado con un vértice no visitado
        for edge in graph.weights:
            if visited[graph.nodes.index(edge.node_a)] ^ visited[graph.nodes.index(edge.node_b)]:
                if edge.cost < min_weight:
                    min_weight = edge.cost
                    min_edge = edge

        if min_edge:
            # Agrega el borde mínimo al árbol de expansión mínima
            mst.append(min_edge)
            # Marca el vértice no visitado como visitado
            if not visited[graph.nodes.index(min_edge.node_a)]:
                visited[graph.nodes.index(min_edge.node_a)] = True
            else:
                visited[graph.nodes.index(min_edge.node_b)] = True

    return mst


def kruskal_algorithm(graph):
    # Ordena los bordes por peso en orden ascendente
    sorted_edges = sorted(graph.weights, key=lambda x: x.cost)

    # Inicializa el árbol de expansión mínima y un conjunto de componentes conectadas
    mst = []
    connected_components = []

    for node in graph.nodes:
        connected_components.append([node])

    # Itera a través de los bordes ordenados
    for edge in sorted_edges:
        component_a = None
        component_b = None

        # Encuentra las componentes conectadas de los vértices finales del borde
        for component in connected_components:
            if edge.node_a in component:
                component_a = component
            if edge.node_b in component:
                component_b = component

        # Si los vértices finales no están en la misma componente, agrega el borde al árbol de expansión mínima
        if component_a != component_b:
            mst.append(edge)
            # Fusiona las dos componentes conectadas en una
            component_a.extend(component_b)
            connected_components.remove(component_b)

    return mst

def print_mst(mst):
    for obj in mst:
        print(obj.node_a.value)


def dijkstra(graph, start_node):
    # Inicializa las distancias, el conjunto de nodos visitados y el camino más corto
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    visited = set()
    shortest_paths = {node: [] for node in graph.nodes}

    # Inicializa la cola de prioridad (MinHeap personalizado)
    priority_queue = gl.MinHeap()
    priority_queue.push((0, start_node))

    while priority_queue.heap:
        current_distance, current_node = priority_queue.pop()

        # Si ya visitamos este nodo, lo ignoramos
        if current_node in visited:
            continue

        # Marca el nodo como visitado
        visited.add(current_node)

        # Actualiza las distancias y el camino más corto a los nodos vecinos
        for edge in graph.weights:
            if edge.node_a == current_node:
                neighbor = edge.node_b
            elif edge.node_b == current_node:
                neighbor = edge.node_a
            else:
                continue

            new_distance = current_distance + edge.cost

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                priority_queue.push((new_distance, neighbor))
                # Actualiza el camino más corto hacia el vecino
                shortest_paths[neighbor] = shortest_paths[current_node] + [edge]

    return distances, shortest_paths


def shortest_distance_to(graph, start_node=1, goal_node=2):
    if isinstance(start_node, int) and isinstance(goal_node, int):
        # Si los argumentos son índices, utiliza la lógica original
        distances, paths = dijkstra(graph, graph.nodes[start_node])
        path_to = paths[graph.nodes[goal_node]]
        distance_to = distances[graph.nodes[goal_node]]
        return path_to, distance_to
    elif isinstance(start_node, str) and isinstance(goal_node, str):
        # Si los argumentos son etiquetas, utiliza la lógica para etiquetas
        start_node = next((node for node in graph.nodes if node.tag == start_node), None)
        goal_node = next((node for node in graph.nodes if node.tag == goal_node), None)

        if start_node is None or goal_node is None:
            return None

        distances, paths = dijkstra(graph, start_node)
        path_to = paths[goal_node]
        distance_to = distances[goal_node]
        return path_to, distance_to
    else:
        # Si los argumentos no son del mismo tipo, retorna None o maneja el caso según tus necesidades
        return None

def draw_edges(lines_g : [], edges, color : str, draw_cost=False, batch=None, width=1):
    lines_g.clear()
    color = color.lstrip("#")
    rgb_color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

    for w in edges:
        w_x = (w.node_a.x + w.node_b.x) / 2
        w_y = (w.node_a.y + w.node_b.y) / 2
        lines_g.append(shapes.Line(w.node_a.x, w.node_a.y, w.node_b.x, w.node_b.y, color=rgb_color, batch=batch, width=width))
        if (draw_cost):
            lines_g.append(shapes.Rectangle(x=w_x-8, y=w_y-8, width=16, height=16, color=(0, 0, 0), batch=batch))
            lines_g.append(pyglet.text.Label(str(w.cost), font_name='Cantarell', font_size=8, x = w_x, y = w_y, anchor_x='center', anchor_y='center', batch=batch))

def draw_graph_pyg(shapes_g = [], graph=None, fill_color : str = '', outline_color : str = '', batch=None, radius = 16):
    shapes_g.clear()
    fill_color = fill_color.lstrip("#")
    outline_color = outline_color.lstrip("#")
    rgb_fcolor = tuple(int(fill_color[i:i+2], 16) for i in (0, 2, 4))
    rgb_ocolor = tuple(int(outline_color[i:i+2], 16) for i in (0, 2, 4))

    for n in graph.nodes:
        shapes_g.append(shapes.Circle(n.x, n.y, radius+2, color=rgb_ocolor, batch=batch))
        shapes_g.append(shapes.Circle(n.x, n.y, radius, color=rgb_fcolor, batch=batch))
        shapes_g.append(pyglet.text.Label(n.tag + ":" + str(n.value), font_name='Agave Nerd Font', font_size=11, x = n.x, y = n.y, anchor_x='center', anchor_y='center', batch=batch))

def re_draw_graph_pyg(shapes_g = [], index=-1, graph=None, fill_color : str = '', outline_color : str = '', batch=None, radius = 16):
    fill_color = fill_color.lstrip("#")
    outline_color = outline_color.lstrip("#")
    rgb_fcolor = tuple(int(fill_color[i:i+2], 16) for i in (0, 2, 4))
    rgb_ocolor = tuple(int(outline_color[i:i+2], 16) for i in (0, 2, 4))

    shapes_g[(index * 3)]     = (shapes.Circle(graph.nodes[index].x, graph.nodes[index].y, radius+2, color=rgb_ocolor, batch=batch))
    shapes_g[(index * 3) + 1]   = (shapes.Circle(graph.nodes[index].x, graph.nodes[index].y, radius, color=rgb_fcolor, batch=batch))
    shapes_g[(index * 3) + 2]   = (pyglet.text.Label(graph.nodes[index].tag + ":" + str(graph.nodes[index].value), font_name='Agave Nerd Font', font_size=11, x = graph.nodes[index].x, y = graph.nodes[index].y, anchor_x='center', anchor_y='center', batch=batch))

class main_window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        config = pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=8)
        super().__init__(*args,config=config, **kwargs)
        self.graph_g = None
        self.batch = None
        self.graph = None
        self.lines_g = None
        self.vp_x = 0
        self.vp_y = 0
        self.vp_z = 0

        self.cur_node = None
        self.node_index = -1

    def on_draw(self):
        self.clear()
        if (self.batch):
                self.batch.draw()

    def on_key_release(self, symbol, modifiers):
        if (symbol == pyglet.window.key.Q):
            pyglet.app.exit()
        if (symbol == pyglet.window.key.D):
            self.vp_x += 10
            self.view = self.view.from_translation((self.vp_x, self.vp_y, 0))
        if (symbol == pyglet.window.key.A):
            self.vp_x -= 10
            self.view = self.view.from_translation((self.vp_x, self.vp_y, 0))

        if (symbol == pyglet.window.key.W):
            self.vp_y += 10
            self.view = self.view.from_translation((self.vp_x, self.vp_y, 0))
        if (symbol == pyglet.window.key.S):
            self.vp_y -= 10
            self.view = self.view.from_translation((self.vp_x, self.vp_y, 0))

    def on_mouse_press(self, x, y, buttons, modifiers):
        if (buttons == pyglet.window.mouse.LEFT):
            for i, n in enumerate(self.graph.nodes):
                distance = gl.math.sqrt(gl.math.pow((x - self.vp_x) - n.x, 2) + gl.math.pow((y - self.vp_y) - n.y, 2))
                if (distance < 20):
                    self.cur_node = n
                    self.node_index = i
            if (self.cur_node):
                self.node_index = self.node_index

    def on_mouse_release(self, x, y, buttons, modifiers):
        if (buttons == pyglet.window.mouse.LEFT):
            self.cur_node = None
            self.node_index = -1
            draw_edges(lines_g=self.lines_g, edges=self.graph.weights, color='#00CFD5', batch=self.batch, draw_cost=True)
            graph_g.update()
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.vp_z += (scroll_y*2)
        print(self.vp_z)
        self.view = self.view.from_translation((self.vp_x, self.vp_y, self.vp_z))


    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if ( buttons == pyglet.window.mouse.MIDDLE or buttons == pyglet.window.mouse.RIGHT):
            self.vp_x += dx
            self.vp_y += dy
            self.view = self.view.from_translation((self.vp_x, self.vp_y, 0))
        if (buttons == pyglet.window.mouse.LEFT):
            if (self.cur_node):
                self.cur_node.x = x - self.vp_x
                self.cur_node.y = y - self.vp_y
                graph_g.nodes[self.node_index].draw()


class node_g:
    def __init__(self, node : gl.node, outline_color : str = '#C40F8F', fill_color : str = '#3B435B', radius : float = 20.0, batch = None):
        fill_color = fill_color.lstrip("#")
        outline_color = outline_color.lstrip("#")
        self.outline_color = tuple(int(outline_color[i:i+2], 16) for i in (0, 2, 4))
        self.fill_color = tuple(int(fill_color[i:i+2], 16) for i in (0, 2, 4))
        self.node = node
        self.batch = batch
        self.tag = node.tag
        self.radius = radius
        self.draw()
    def draw(self):
        self.outline_s = shapes.Circle(self.node.x, self.node.y, self.radius+2, color=self.outline_color, batch=self.batch, group=pyglet.graphics.Group(order=0))
        self.fill_s = shapes.Circle(self.node.x, self.node.y, self.radius, color=self.fill_color, batch=self.batch, group=pyglet.graphics.Group(order=1))
        self.text = pyglet.text.Label(self.node.tag + ":" + str(self.node.value), font_name='Agave Nerd Font', font_size=11, x = self.node.x, y = self.node.y, anchor_x='center', anchor_y='center', batch=self.batch, group=pyglet.graphics.Group(order=2))

class graph_g:
    def __init__(self, graph : gl.random_graph, batch = None):
        self.graph = graph
        self.batch = batch
        self.nodes = []
        self.edges = []


    def prepare(self):
        for n in self.graph.nodes:
            self.nodes.append(node_g(node=n, batch=self.batch))
    def update(self):
        for n in self.nodes:
            n.draw

if __name__ == '__main__':

    seed = gl.random.randint(0, 1000000000)
    gl.random.seed(seed)
    print(seed)
    graph = gl.random_graph()
    graph.set_canvas_size(1000, 1000)
    graph.set_structure_size(size=15)
    graph.set_nodes_max_val(99)
    graph.set_weights_max_val(20)
    graph.preparing()
    graph.build_random(full_connected=True, density=20)

    gl.force_directed_layout_weight(graph, iterations=1000, k_repulsion=1200.0, k_attraction_base=0.005)
    mst = kruskal_algorithm(graph)
    path_to, distance_to = shortest_distance_to(graph, "A", "B")
    print("distance: " + str(distance_to))

    batch = pyglet.graphics.Batch()
    nodes_g = []
    lines_g = []
    nodes_list = []
    window = main_window(width=1920, height=1080, resizable=True)
    graph_g = graph_g(graph, batch)
    graph_g.prepare()
    window.graph_g = graph_g
    draw_edges(lines_g=lines_g, edges=graph.weights, color='#00CFD5', batch=batch, draw_cost=True)
    #draw_edges(lines_g=lines_w, edges=path_to, color='#90FF09', batch=batch, draw_cost=True, width=3)
    #draw_edges(lines_g=shapes_g, edges=mst, color='#BA5337', batch=batchs[0], draw_cost=False, width=3)
    window.batch = batch
    window.graph = graph
    window.lines_g = lines_g

    pyglet.app.run()





