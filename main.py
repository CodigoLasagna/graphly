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


class main_window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        config = pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=4)
        super().__init__(*args, config=config, **kwargs)
        self.graph_g = None
        self.batch = None
        self.graph = None
        self.vp_x = 0
        self.vp_y = 0
        self.vp_z = 0
        
        bkgcolor_h = "#1e2935"
        bkgcolor_h = bkgcolor_h.strip('#')
        self.bkg_c = tuple(int(bkgcolor_h[i:i+2], 16) for i in (0, 2, 4))
        #pyglet.gl.glClearColor(self.bkg_c[0]/255, self.bkg_c[1]/255, self.bkg_c[2]/255, 1)

        self.cur_node = None
        self.node_index = -1
        self.cur_edges = []

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
                for i, e in enumerate(self.graph.weights):
                    if (e.node_a == self.cur_node or e.node_b == self.cur_node):
                        self.cur_edges.append(self.graph_g.edges[i])

    def on_mouse_release(self, x, y, buttons, modifiers):
        if (buttons == pyglet.window.mouse.LEFT):
            self.cur_node = None
            self.node_index = -1
            self.cur_edges.clear()
            #draw_edges(lines_g=self.lines_g, edges=self.graph.weights, color='#00CFD5', batch=self.batch, draw_cost=True)
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
                if (len(self.cur_edges) > 0):
                    for c_e in self.cur_edges:
                        c_e.draw()


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
        self.outline_s = shapes.Circle(self.node.x, self.node.y, self.radius+2, color=self.outline_color, batch=self.batch, group=pyglet.graphics.Group(order=1))
        self.fill_s = shapes.Circle(self.node.x, self.node.y, self.radius, color=self.fill_color, batch=self.batch, group=pyglet.graphics.Group(order=2))
        self.text = pyglet.text.Label(self.node.tag + ":" + str(self.node.value), font_name='Agave Nerd Font', font_size=11, x = self.node.x, y = self.node.y, anchor_x='center', anchor_y='center', batch=self.batch, group=pyglet.graphics.Group(order=3))

class edge_g:
    def __init__(self, color : str = '#00CFD5', width = 1, edge : gl.weight = None, batch = None, show_cost : bool = False):
        self.color_h = color
        self.color_hs = self.color_h.lstrip("#")
        self.color = tuple(int(self.color_hs[i:i+2], 16) for i in (0, 2, 4))
        self.width = width
        self.edge = edge
        self.batch = batch
        self.show_cost = show_cost
        self.cost_bkg = None
        self.label = None
        self.draw()

    def draw(self):
        self.color_hs = self.color_h.lstrip("#")
        self.color = tuple(int(self.color_hs[i:i+2], 16) for i in (0, 2, 4))
        self.line_s = shapes.Line(x=self.edge.node_a.x, y=self.edge.node_a.y, x2=self.edge.node_b.x, y2=self.edge.node_b.y, color=self.color, batch=self.batch, width=self.width, group=pyglet.graphics.Group(order=0))
        if (self.show_cost):
            w_x = (self.edge.node_a.x + self.edge.node_b.x)/2
            w_y = (self.edge.node_a.y + self.edge.node_b.y)/2
            self.cost_bkg = shapes.Rectangle(x=w_x-8, y=w_y-8, width=16, height=16, color=(0, 0, 0), batch=self.batch, group=pyglet.graphics.Group(order=1))
            self.label = pyglet.text.Label(str(self.edge.cost), font_name='Cantarell', font_size=8, x = w_x, y = w_y, anchor_x='center', anchor_y='center', batch=self.batch, group=pyglet.graphics.Group(order=1))
        else:
            self.cost_bkg = None
            self.label = None

class graph_g:
    def __init__(self, graph : gl.random_graph, batch = None):
        self.graph = graph
        self.batch = batch
        self.nodes = []
        self.edges = []


    def prepare(self):
        for n in self.graph.nodes:
            self.nodes.append(node_g(node=n, batch=self.batch))
        for w in self.graph.weights:
            self.edges.append(edge_g(edge=w, batch=self.batch))
    def update(self):
        for n in self.nodes:
            n.draw()
        for w in self.edges:
            w.draw()

    def update_weights(self, n_edges, color='#00CFD5', width=1, show_cost=False):
        for w in self.edges:
            for w_n in n_edges:
                if (w.edge == w_n):
                    w.color_h = color
                    w.width = width
                    w.show_cost = show_cost
        self.update()

if __name__ == '__main__':

    seed = gl.random.randint(0, 1000000000)
    #seed = 20
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
    path_to, distance_to = shortest_distance_to(graph, "A", "D")
    print("distance: " + str(distance_to))

    batch = pyglet.graphics.Batch()
    nodes_list = []
    window = main_window(width=1920, height=1080, resizable=True)
    graph_g = graph_g(graph, batch)
    graph_g.prepare()
    window.graph_g = graph_g
    #graph_g.update_weights(mst, '#BA5337', 2)
    #graph_g.update_weights(graph.weights, '#00CFD5', 1, True)
    graph_g.update_weights(path_to, '#90FF09', 3, True)
    window.batch = batch
    window.graph = graph

    pyglet.app.run()





