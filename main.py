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

window = pyglet.window.Window()

label = pyglet.text.Label('Hello there',
                          font_name='BlexMono Nerd Font',
                          font_size=36,
                          x=window.width//2,
                          y=window.height//2,
                          anchor_x='center',
                          anchor_y='center')

batch = pyglet.graphics.Batch()

nodes_c = []
lines_w = []
label_t = []

def draw_graph_pyg(graph):

    for w in graph.weights:
        lines_w.append(shapes.Line(w.node_a.x, w.node_a.y, w.node_b.x, w.node_b.y, color=(10, 50, 170), batch=batch))
        label_t.append(pyglet.text.Label(str(w.cost), font_name='BlexMono Nerd Font', font_size=12, x = (w.node_a.x + w.node_b.x) / 2, y = (w.node_a.y + w.node_b.y) / 2, anchor_x='center', anchor_y='center', batch=batch))
    for n in graph.nodes:
        nodes_c.append(shapes.Circle(n.x, n.y, 26, color=(129, 128, 96), batch=batch))
        nodes_c.append(shapes.Circle(n.x, n.y, 24, color=(20, 128, 96), batch=batch))
        label_t.append(pyglet.text.Label(n.tag + ":" + str(n.value), font_name='BlexMono Nerd Font', font_size=12, x = n.x, y = n.y, anchor_x='center', anchor_y='center', batch=batch))


@window.event
def on_draw():
    window.clear()
    batch.draw()
    #label.draw()


if __name__ == '__main__':
    #Seeds interesantes
    #s:19 nquant:6
    seed = gl.random.randint(0, 1000000000)
    #seed = 409481041
    #seed = 780144336
    #seed = 666961458
    #seed = 837479247
    #seed = 784109775
    #seed = 66913871
    seed = 51358019
    gl.random.seed(seed)
    print(seed)
    graph = gl.random_graph()
    graph.set_canvas_size(1920, 1080)
    graph.set_structure_size(size=15)
    graph.set_nodes_max_val(99)
    graph.set_weights_max_val(20)
    graph.preparing()
    graph.build_random(full_connected=True, density=25)

    gl.force_directed_layout_weight(graph, iterations=1000, k_repulsion=3000.0, k_attraction_base=0.005)
    mst = kruskal_algorithm(graph)
    path_to, distance_to = shortest_distance_to(graph, "G", "H")
    print("distance: " + str(distance_to))

    canvas = gl.canvas_drawing()
    canvas.set_size(graph)
    canvas.set_canvas_bkg('#1B232B')
    canvas.draw_weights(graph.weights, '#00CFD5', 1, True)
    canvas.draw_weights(path_to, '#90FF09', 4, True)
    #canvas.draw_weights(mst, '#BA5337', 3, False)
    canvas.draw_graph(graph, radius=40, filler='#3B435B', text_c='#9f9f9f', node_outline='#24947F')

    draw_graph_pyg(graph)
    pyglet.app.run()



