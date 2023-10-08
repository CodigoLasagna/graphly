import graphlib as gl


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

if __name__ == '__main__':
    #Seeds interesantes
    #s:19 nquant:6
    gl.random.seed(38)
    graph = gl.random_graph()
    graph.set_canvas_size(1920, 1080)
    graph.set_structure_size(size=10)
    graph.set_nodes_params(10)
    graph.preparing()
    graph.build_random(full_connected=True, density=20)

    gl.force_directed_layout_weight(graph, iterations=1000, k_repulsion=2000.0, k_attraction_base=0.01)
    mst = kruskal_algorithm(graph)
    canvas = gl.canvas_drawing()
    canvas.set_size(graph)
    canvas.set_canvas_bkg('#1B232B')
    canvas.draw_mst(graph.weights, '#00CFD5', 1, True)
    canvas.draw_mst(mst, '#BA5337', 3, False)
    canvas.draw_graph(graph, radius=40, filler='#3B435B', text_c='#9f9f9f', node_outline='#24947F')
    #canvas.draw_mst(graph.weights)

