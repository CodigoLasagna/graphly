import random
import math
from PIL import Image, ImageDraw, ImageFont


def gen_random_number(size):
    return random.randint(0, size)


class node:
    def __init__(self, n_size, tag, c_width, c_height):
        self.value = gen_random_number(n_size)
        self.tag = tag
        self.neighbors = []
        self.sep_factor = 300
        self.x = random.randint((c_width/2)-self.sep_factor, (c_width/2)+self.sep_factor)
        self.y = random.randint((c_height/2)-self.sep_factor, (c_height/2)+self.sep_factor)


class weight:
    def __init__(self, node_a, node_b):
        self.cost = random.randint(0, 10)
        self.node_a = node_a
        self.node_b = node_b


class random_graph:
    def __init__(self):
        self.tag = ""
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.tag_count = 0
        self.nodes = []
        self.weights = []
        self.structure_size = 10
        self.node_n_size = 10
        self.c_width = 900
        self.c_height = 900
        self.density = 100

    def set_canvas_size(self, width, height):
        self.c_width = width
        self.c_height = height

    def set_structure_size(self, size):
        self.structure_size = size

    def set_nodes_params(self, n_size):
        self.node_n_size = n_size

    def generate_next_tag(self):
        result = ""
        quotient = self.tag_count
        while quotient >= 0:
            quotient, remainder = divmod(quotient, len(self.letters))
            result = self.letters[remainder] + result
            quotient -= 1
        return result

    def preparing(self):
        for i in range(self.structure_size):
            self.tag = self.generate_next_tag()
            self.nodes.append(node(self.node_n_size, self.tag, self.c_width, self.c_height))
            self.tag_count += 1

    def build_random(self, full_connected = True, density = 100):
        st = self.structure_size
        existing_weights = set()
        main_tree = []
        main_tree.append(random.choice(self.nodes))
        self.density = (density/100)

        for node in self.nodes:
            for i in range(random.randint(0, math.floor((st-1)*self.density))):
                new_node = self.nodes[random.randint(0, st-1)]
                while ((new_node in node.neighbors) or node == new_node):
                    new_node = self.nodes[random.randint(0, st-1)]

                if (new_node, node) not in existing_weights and (node, new_node) not in existing_weights:
                    node.neighbors.append(new_node)
                    if (new_node in main_tree):
                        if (node not in main_tree):
                            main_tree.append(node)
                    self.weights.append(weight(node, new_node))
                    existing_weights.add((node, new_node))
                    existing_weights.add((new_node, node))

        if (full_connected):
            for node in self.nodes:
                if (node not in main_tree):
                    self.weights.append(weight(node, random.choice(main_tree)))
                    main_tree.append(node)


    def print_graph(self):
        result = ""
        for node in self.nodes:
            # FIX: this is broken
            result = node.tag + " - VAL["+str(node.value)+"] - N["
            result += str(node.neighbor.tag) + "]"
            print(result)

    def print_weitghts(self):
        result = ""
        for w in self.weights:
            result = "N1["+str(w.node_a.value)+"]["+w.node_a.tag+"] - "
            result += "N2["+str(w.node_b.value)+"]["+w.node_b.tag+"] "
            result += "w["+str(w.cost)+"] "
            print(result)

class canvas_drawing:
    def __init__(self):
        self.height = 600
        self.width = 800
        self.image = Image.new('RGB', (self.width, self.height),'white')
        self.draw = ImageDraw.Draw(self.image)
        self.m_font = ImageFont.truetype("CodeNext.otf", 14)

    def set_size(self, graph):
        self.width = graph.c_width
        self.height = graph.c_height

    def set_canvas_bkg(self, bkg):
        self.image = Image.new('RGB', (self.width, self.height),bkg)
        self.draw = ImageDraw.Draw(self.image)

    def draw_mst(self, weights, color, line_w, draw_w):
        for w in weights:
            self.draw.line([(w.node_a.x, w.node_a.y), (w.node_b.x, w.node_b.y)], fill=color, width=line_w)

            if (draw_w == True):
                self.draw.text([(w.node_a.x+w.node_b.x)/2, (w.node_a.y+w.node_b.y)/2 - 20], str(w.cost), fill='#8f8f8f', font=self.m_font)

        self.image.save('graph.png')

    def draw_graph(self, graph, radius, filler, text_c, node_outline):
        for node in graph.nodes:
            self.draw.ellipse([node.x-(radius/2), node.y-(radius/2), node.x+(radius/2), node.y+(radius/2)], fill=filler,outline=node_outline, width=2)

            self.draw.text([node.x-(radius/4), node.y-(radius/4)], node.tag +":"+ str(node.value), fill=text_c, font=self.m_font)

        self.image.save('graph.png')


def calculate_repulsion(node1, node2, k_repulsion):
    dx = node1.x - node2.x
    dy = node1.y - node2.y
    distance = math.sqrt(dx**2 + dy**2)
    if distance == 0:
        return (0, 0)
    force = k_repulsion / distance**2
    force_x = force * dx / distance
    force_y = force * dy / distance
    return (force_x, force_y)

def calculate_attraction(node1, node2, k_attraction):
    dx = node2.x - node1.x
    dy = node2.y - node1.y
    distance = math.sqrt(dx**2 + dy**2)
    force = k_attraction * distance
    force_x = force * dx / distance
    force_y = force * dy / distance
    return (force_x, force_y)

def force_directed_layout(graph, iterations=100, k_repulsion=1.0, k_attraction=0.1):
    for _ in range(iterations):
        for node in graph.nodes:
            node.force_x = 0
            node.force_y = 0

        for i, node1 in enumerate(graph.nodes):
            for j, node2 in enumerate(graph.nodes):
                if i < j:
                    repulsion_x, repulsion_y = calculate_repulsion(node1, node2, k_repulsion)
                    node1.force_x += repulsion_x
                    node1.force_y += repulsion_y
                    node2.force_x -= repulsion_x
                    node2.force_y -= repulsion_y

        for weight in graph.weights:
            attraction_x, attraction_y = calculate_attraction(weight.node_a, weight.node_b, k_attraction)
            weight.node_a.force_x += attraction_x
            weight.node_a.force_y += attraction_y
            weight.node_b.force_x -= attraction_x
            weight.node_b.force_y -= attraction_y

        for node in graph.nodes:
            node.x += node.force_x
            node.y += node.force_y

def force_directed_layout_weight(graph, iterations=100, k_repulsion=1.0, k_attraction_base=0.1):
    for _ in range(iterations):
        for node in graph.nodes:
            node.force_x = 0
            node.force_y = 0

        for i, node1 in enumerate(graph.nodes):
            for j, node2 in enumerate(graph.nodes):
                if i < j:
                    repulsion_x, repulsion_y = calculate_repulsion(node1, node2, k_repulsion * len(graph.nodes))
                    node1.force_x += repulsion_x
                    node1.force_y += repulsion_y
                    node2.force_x -= repulsion_x
                    node2.force_y -= repulsion_y

        for weight in graph.weights:
            attraction_x, attraction_y = calculate_attraction(weight.node_a, weight.node_b, k_attraction_base * weight.cost * (graph.density))
            weight.node_a.force_x += attraction_x
            weight.node_a.force_y += attraction_y
            weight.node_b.force_x -= attraction_x
            weight.node_b.force_y -= attraction_y

        for node in graph.nodes:
            node.x += node.force_x
            node.y += node.force_y
