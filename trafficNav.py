import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
import heapq
import math
import json
from typing import Dict, Any, Tuple, List


class Graph:
    def __init__(self):
        self.adj: Dict[Any, List[Tuple[Any, float]]] = {}
        self.coords: Dict[Any, Tuple[float, float]] = {}
        self.edges: Dict[Tuple[Any, Any], float] = {}

    def add_node(self, node, x: float, y: float):
        if node not in self.adj:
            self.adj[node] = []
            self.coords[node] = (x, y)

    def remove_node(self, node):
        if node not in self.adj:
            return
        
        for neighbor, _ in self.adj[node]:
            self.adj[neighbor] = [(n, w) for n, w in self.adj[neighbor] if n != node]
            if (neighbor, node) in self.edges:
                del self.edges[(neighbor, node)]
        
        for other_node in list(self.adj.keys()):
            if other_node != node:
                self.adj[other_node] = [(n, w) for n, w in self.adj[other_node] if n != node]
        
        edges_to_remove = [edge for edge in self.edges if node in edge]
        for edge in edges_to_remove:
            del self.edges[edge]
        
        del self.adj[node]
        del self.coords[node]

    def add_edge(self, u, v, weight: float, bidirectional: bool = True):
        self.adj.setdefault(u, []).append((v, weight))
        self.edges[(u, v)] = weight
        if bidirectional:
            self.adj.setdefault(v, []).append((u, weight))
            self.edges[(v, u)] = weight

    def neighbors(self, node):
        """Return neighbors with their current edge weights"""
        return self.adj.get(node, [])

    def update_edge_weight(self, u, v, new_weight: float):
        # Update edge u -> v in adjacency list
        for i, (neighbor, _) in enumerate(self.adj[u]):
            if neighbor == v:
                self.adj[u][i] = (v, new_weight)
                self.edges[(u, v)] = new_weight
                break
        
        # Update edge v -> u in adjacency list (bidirectional)
        for i, (neighbor, _) in enumerate(self.adj.get(v, [])):
            if neighbor == u:
                self.adj[v][i] = (u, new_weight)
                self.edges[(v, u)] = new_weight
                break

    def remove_edge(self, u, v):
        """Remove edge between nodes u and v"""
        self.adj[u] = [(n, w) for n, w in self.adj[u] if n != v]
        self.adj[v] = [(n, w) for n, w in self.adj[v] if n != u]
        if (u, v) in self.edges:
            del self.edges[(u, v)]
        if (v, u) in self.edges:
            del self.edges[(v, u)]

    def to_dict(self):
        """Export graph to dictionary for saving"""
        return {
            'nodes': {str(k): list(v) for k, v in self.coords.items()},
            'edges': {f"{u},{v}": w for (u, v), w in self.edges.items()}
        }
    
    def from_dict(self, data):
        """Import graph from dictionary"""
        self.adj.clear()
        self.coords.clear()
        self.edges.clear()
        
        for node, (x, y) in data['nodes'].items():
            self.add_node(node, x, y)
        
        processed = set()
        for edge_key, weight in data['edges'].items():
            u, v = edge_key.split(',')
            if (u, v) not in processed and (v, u) not in processed:
                self.add_edge(u, v, weight)
                processed.add((u, v))


def dijkstra(graph: Graph, start, target) -> Tuple[float, List, List]:
    dist = {node: math.inf for node in graph.adj}
    dist[start] = 0.0
    parent = {}
    visited = []
    pq = [(0.0, start)]
    
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]: 
            continue
        visited.append(node)
        if node == target: 
            break
        
        for nbr, w in graph.neighbors(node):
            nd = d + w
            if nd < dist[nbr]:
                dist[nbr] = nd
                parent[nbr] = node
                heapq.heappush(pq, (nd, nbr))
    
    if target not in parent and start != target:
        return math.inf, [], visited
    
    path = []
    cur = target
    while cur in parent:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    
    return dist[target], path, visited


def a_star(graph: Graph, start, target) -> Tuple[float, List, List]:
    def h(n):
        (x1, y1) = graph.coords[n]
        (x2, y2) = graph.coords[target]
        return math.hypot(x1 - x2, y1 - y2)

    g_score = {node: math.inf for node in graph.adj}
    g_score[start] = 0.0
    f_score = {node: math.inf for node in graph.adj}
    f_score[start] = h(start)
    
    pq = [(f_score[start], start)]
    parent = {}
    visited = []

    while pq:
        _, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.append(node)
        
        if node == target:
            path = []
            cur = target
            while cur in parent:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            path.reverse()
            return g_score[target], path, visited

        for nbr, w in graph.neighbors(node):
            tentative_g = g_score[node] + w
            if tentative_g < g_score[nbr]:
                parent[nbr] = node
                g_score[nbr] = tentative_g
                f_score[nbr] = tentative_g + h(nbr)
                heapq.heappush(pq, (f_score[nbr], nbr))
    
    return math.inf, [], visited


def build_sample_graph() -> Graph:
    g = Graph()
    nodes = {'A': (100, 80), 'B': (200, 100), 'C': (320, 80), 'D': (100, 200),
             'E': (200, 200), 'F': (320, 200), 'G': (100, 320), 'H': (200, 320),
             'I': (320, 320), 'J': (440, 200)}
    for n, (x, y) in nodes.items():
        g.add_node(n, x, y)
    
    edges = [('A','B',1.0), ('B','C',1.0), ('A','D',1.2), ('B','E',1.1), ('C','F',1.2),
             ('D','E',1.0), ('E','F',1.0), ('D','G',1.3), ('E','H',1.0), ('F','I',1.3),
             ('G','H',1.0), ('H','I',1.0), ('F','J',0.9), ('C','J',1.6)]
    for u, v, w in edges:
        g.add_edge(u, v, w)
    return g


class TrafficNavApp(tk.Tk):
    def __init__(self, graph: Graph):
        super().__init__()
        self.graph = graph
        self.path = []
        self.visited_nodes = []
        self.total_cost = 0.0
        self.mode = tk.StringVar(value="navigate")  
        self.selected_node = None 
        self.selected_edge_start = None
        self.node_counter = 0
        self.show_visited = tk.BooleanVar(value=True)

        self.title("Enhanced Traffic Navigator Pro")
        self.geometry("1100x700")
        self.configure(bg='#f0f0f0')

        self.create_menu()
        self.create_ui()
        self.draw_map()

    def show_about(self):
        messagebox.showinfo(
            "About Traffic Navigator",
            "Enhanced Traffic Navigator Pro\n"
            "Version 1.0\n\n"
            "Features:\n"
            "• Interactive graph editing\n"
            "• Real-time path finding\n"
            "• Traffic simulation\n"
            "• A* and Dijkstra algorithms"
        )

    def create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Graph", command=self.new_graph)
        file_menu.add_command(label="Load Sample", command=self.load_sample)
        file_menu.add_separator()
        file_menu.add_command(label="Save Graph", command=self.save_graph)
        file_menu.add_command(label="Load Graph", command=self.load_graph)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Visited Nodes", variable=self.show_visited, 
                                   command=self.draw_map)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_ui(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(main_frame, relief=tk.RIDGE, borderwidth=2)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=1, relief=tk.SOLID)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_hover)

        title = ttk.Label(control_frame, text="Traffic Navigator", 
                         font=("Helvetica", 16, "bold"), foreground="#2c3e50")
        title.pack(pady=10)

        mode_frame = ttk.LabelFrame(control_frame, text="Mode", padding="10")
        mode_frame.pack(fill='x', padx=10, pady=5)
        
        modes = [
            ("Navigate", "navigate"),
            ("Add Node", "add_node"),
            ("Remove Node", "remove_node"),
            ("Add Edge", "add_edge"),
            ("Edit Edge", "edit_edge"),
            ("Remove Edge", "remove_edge")
        ]
        
        for text, value in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode, 
                          value=value, command=self.change_mode).pack(anchor='w', pady=2)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        nav_frame = ttk.LabelFrame(control_frame, text="Navigation", padding="10")
        nav_frame.pack(fill='x', padx=10, pady=5)

        nodes = sorted(self.graph.adj.keys())
        self.start_node = tk.StringVar(value=nodes[0] if nodes else "")
        self.end_node = tk.StringVar(value=nodes[-1] if nodes else "")
        
        ttk.Label(nav_frame, text="Start:").pack(anchor='w', pady=(5, 0))
        self.start_combo = ttk.Combobox(nav_frame, textvariable=self.start_node, state="readonly")
        self.start_combo['values'] = nodes
        self.start_combo.pack(fill='x')
        
        ttk.Label(nav_frame, text="Destination:").pack(anchor='w', pady=(10, 0))
        self.end_combo = ttk.Combobox(nav_frame, textvariable=self.end_node, state="readonly")
        self.end_combo['values'] = nodes
        self.end_combo.pack(fill='x')

        self.algo = tk.StringVar(value="A*")
        ttk.Label(nav_frame, text="Algorithm:").pack(anchor='w', pady=(10, 0))
        algo_combo = ttk.Combobox(nav_frame, textvariable=self.algo, 
                                 values=["A*", "Dijkstra"], state="readonly")
        algo_combo.pack(fill='x')

        btn_frame = ttk.Frame(nav_frame)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="Find Path", command=self.find_path).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Clear Path", command=self.clear_path).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Report Traffic", command=self.report_traffic).pack(fill='x', pady=2)

        info_frame = ttk.LabelFrame(control_frame, text="Information", padding="10")
        info_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.status_label = ttk.Label(info_frame, text="Ready", 
                                      wraplength=200, font=("Helvetica", 9, "italic"),
                                      foreground="#7f8c8d")
        self.status_label.pack(pady=5, fill='x')
        
        self.result_label = ttk.Label(info_frame, text="", wraplength=200, 
                                      justify='left', foreground="#2c3e50")
        self.result_label.pack(pady=5, fill='both', expand=True)

        stats_text = f"Nodes: {len(self.graph.adj)} | Edges: {len(self.graph.edges)//2}"
        self.stats_label = ttk.Label(control_frame, text=stats_text, 
                                     font=("Helvetica", 8), foreground="#95a5a6")
        self.stats_label.pack(side=tk.BOTTOM, pady=5)

    def update_node_lists(self):
        nodes = sorted(self.graph.adj.keys())
        
        if self.start_node.get() not in nodes:
            self.start_node.set(nodes[0] if nodes else "")
        if self.end_node.get() not in nodes:
            self.end_node.set(nodes[-1] if nodes else "")
        
        self.start_combo['values'] = nodes
        self.end_combo['values'] = nodes
        
        stats = f"Nodes: {len(self.graph.adj)} | Edges: {len(self.graph.edges)//2}"
        self.stats_label.config(text=stats)

    def change_mode(self):
        mode = self.mode.get()
        self.selected_node = None
        self.selected_edge_start = None
        
        messages = {
            "navigate": "Navigate mode: Use controls to find paths",
            "add_node": "Click on canvas to add a new node",
            "remove_node": "Click on a node to remove it",
            "add_edge": "Click two nodes to connect them",
            "edit_edge": "Click on an edge to edit its weight",
            "remove_edge": "Click on an edge to remove it"
        }
        
        self.status_label.config(text=messages.get(mode, "Ready"))
        self.draw_map()

    def on_canvas_hover(self, event):
        node = self.find_node_at(event.x, event.y)
        if node:
            self.canvas.config(cursor="hand2")
        else:
            self.canvas.config(cursor="")

    def on_canvas_click(self, event):
        mode = self.mode.get()
        
        if mode == "add_node":
            self.add_node_at(event.x, event.y)
        elif mode == "remove_node":
            node = self.find_node_at(event.x, event.y)
            if node:
                self.remove_node(node)
        elif mode == "add_edge":
            node = self.find_node_at(event.x, event.y)
            if node:
                self.select_node_for_edge(node)
        elif mode == "edit_edge":
            edge = self.find_edge_at(event.x, event.y)
            if edge:
                self.edit_edge(edge)
        elif mode == "remove_edge":
            edge = self.find_edge_at(event.x, event.y)
            if edge:
                self.remove_edge_action(edge)

    def add_node_at(self, x, y):
        while True:
            self.node_counter += 1
            node_name = f"N{self.node_counter}"
            if node_name not in self.graph.adj:
                break
        
        self.graph.add_node(node_name, x, y)
        self.update_node_lists()
        self.draw_map()
        self.status_label.config(text=f"Node '{node_name}' added at ({x}, {y})")

    def remove_node(self, node):
        confirm = messagebox.askyesno("Confirm Removal", 
                                      f"Remove node '{node}' and all its connections?")
        if confirm:
            self.graph.remove_node(node)
            self.path = []
            self.visited_nodes = []
            self.update_node_lists()
            self.draw_map()
            self.status_label.config(text=f"Node '{node}' removed")

    def select_node_for_edge(self, node):
        if self.selected_edge_start is None:
            self.selected_edge_start = node
            self.status_label.config(text=f"First node: {node}\nClick second node")
            self.draw_map()
        elif self.selected_edge_start == node:
            messagebox.showwarning("Same Node", "Please select a different node")
        else:
            if any(n == node for n, _ in self.graph.neighbors(self.selected_edge_start)):
                messagebox.showwarning("Edge Exists", 
                                      f"Edge between {self.selected_edge_start} and {node} already exists")
            else:
                self.create_edge_dialog(self.selected_edge_start, node)
            self.selected_edge_start = None
            self.status_label.config(text="Click two nodes to connect them")
            self.draw_map()

    def create_edge_dialog(self, node1, node2):
        weight = simpledialog.askfloat("Edge Weight", 
                                       f"Enter weight for edge {node1} ↔ {node2}:",
                                       minvalue=0.1, initialvalue=1.0)
        if weight:
            self.graph.add_edge(node1, node2, weight)
            self.update_node_lists()
            self.draw_map()
            self.status_label.config(text=f"Edge added: {node1} ↔ {node2} (weight: {weight})")

    def find_node_at(self, x, y, radius=18):
        for node, (nx, ny) in self.graph.coords.items():
            if math.hypot(x - nx, y - ny) <= radius:
                return node
        return None

    def find_edge_at(self, x, y, threshold=10):
        """Find edge near click position"""
        for (u, v) in self.graph.edges.keys():
            if u > v: continue
            x1, y1 = self.graph.coords[u]
            x2, y2 = self.graph.coords[v]
            
            dist = self.point_to_line_distance(x, y, x1, y1, x2, y2)
            if dist < threshold:
                return (u, v)
        return None

    def point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point to line segment"""
        line_len = math.hypot(x2 - x1, y2 - y1)
        if line_len == 0:
            return math.hypot(px - x1, py - y1)
        
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_len ** 2)))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        return math.hypot(px - proj_x, py - proj_y)

    def edit_edge(self, edge):
        u, v = edge
        current_weight = self.graph.edges[(u, v)]
        
        new_weight = simpledialog.askfloat("Edit Edge Weight", 
                                          f"Current weight: {current_weight}\nEnter new weight:",
                                          minvalue=0.1, initialvalue=current_weight)
        if new_weight:
            self.graph.update_edge_weight(u, v, new_weight)
            self.draw_map()
            self.status_label.config(text=f"Edge {u} ↔ {v} updated to weight {new_weight}")

    def remove_edge_action(self, edge):
        u, v = edge
        confirm = messagebox.askyesno("Confirm Removal", 
                                      f"Remove edge between {u} and {v}?")
        if confirm:
            self.graph.remove_edge(u, v)
            self.path = []
            self.visited_nodes = []
            self.draw_map()
            self.update_node_lists()
            self.status_label.config(text=f"Edge {u} ↔ {v} removed")

    def draw_map(self):
        self.canvas.delete("all")
        node_radius = 18
        
        if self.show_visited.get() and self.visited_nodes:
            for node in self.visited_nodes:
                if node not in self.path:
                    x, y = self.graph.coords[node]
                    self.canvas.create_oval(x-node_radius-5, y-node_radius-5, 
                                          x+node_radius+5, y+node_radius+5, 
                                          fill="#e8f4f8", outline="#3498db", 
                                          width=1, dash=(2, 2), tags="visited")
        
        for (u, v), weight in self.graph.edges.items():
            if u > v: continue
            x1, y1 = self.graph.coords[u]
            x2, y2 = self.graph.coords[v]
            
            color = "gray"
            width = 2
            
            if self.selected_edge_start == u or self.selected_edge_start == v:
                color = "#f39c12"
                width = 3
            
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, 
                                   tags="edge", smooth=True)
            
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            self.canvas.create_oval(mid_x-10, mid_y-10, mid_x+10, mid_y+10,
                                   fill="white", outline=color, tags="weight_bg")
            self.canvas.create_text(mid_x, mid_y, text=f"{weight:.1f}", 
                                   fill="#2c3e50", font=("Helvetica", 9, "bold"))

        if self.path:
            for i in range(len(self.path) - 1):
                u, v = self.path[i], self.path[i+1]
                x1, y1 = self.graph.coords[u]
                x2, y2 = self.graph.coords[v]
                self.canvas.create_line(x1, y1, x2, y2, fill="#e74c3c", 
                                       width=6, tags="path", smooth=True,
                                       capstyle=tk.ROUND)

        for node, (x, y) in self.graph.coords.items():
            if self.selected_edge_start == node:
                fill_color = "#f39c12"
                outline_color = "#d68910"
            elif self.path and node == self.path[0]:
                fill_color = "#2ecc71"
                outline_color = "#27ae60"
            elif self.path and node == self.path[-1]:
                fill_color = "#e74c3c"
                outline_color = "#c0392b"
            elif node in self.path:
                fill_color = "#3498db"
                outline_color = "#2980b9"
            else:
                fill_color = "#ecf0f1"
                outline_color = "#95a5a6"
            
            self.canvas.create_oval(x-node_radius, y-node_radius, 
                                   x+node_radius, y+node_radius, 
                                   fill=fill_color, outline=outline_color, 
                                   width=3, tags="node")
            self.canvas.create_text(x, y, text=node, 
                                   font=("Helvetica", 11, "bold"),
                                   fill="#2c3e50")
            
    def find_path(self):
        start = self.start_node.get()
        end = self.end_node.get()
        
        if not start or not end:
            messagebox.showwarning("Missing Selection", "Please select start and destination")
            return
        
        if start == end:
            self.path = [start]
            self.total_cost = 0
            self.visited_nodes = []
        else:
            if self.algo.get() == "A*":
                cost, path, visited = a_star(self.graph, start, end)
            else:
                cost, path, visited = dijkstra(self.graph, start, end)
            
            self.total_cost = cost
            self.path = path
            self.visited_nodes = visited
        
        self.update_results()
        self.draw_map()
        
    def update_results(self):
        if not self.path or self.total_cost == math.inf:
            self.result_label.config(text=f"❌ No path found\n\nFrom: {self.start_node.get()}\nTo: {self.end_node.get()}")
        else:
            path_str = " → ".join(self.path)
            visited_count = len(self.visited_nodes)
            result_text = (f"✓ {self.algo.get()} Algorithm\n\n"
                          f"Total Cost: {self.total_cost:.2f}\n"
                          f"Nodes in path: {len(self.path)}\n"
                          f"Nodes explored: {visited_count}\n\n"
                          f"Path:\n{path_str}")
            self.result_label.config(text=result_text)

    def clear_path(self):
        self.path = []
        self.visited_nodes = []
        self.total_cost = 0
        self.result_label.config(text="")
        self.draw_map()
        self.status_label.config(text="Path cleared")

    def report_traffic(self):
        if not self.graph.edges:
            messagebox.showwarning("No Edges", "No edges available")
            return
            
        road_choices = sorted([f"{u} → {v}" for u, v in self.graph.edges.keys()])
        dialog = TrafficDialog(self, title="Report Traffic", road_options=road_choices)
        
        if dialog.result:
            road, delay_str = dialog.result
            u, v = road.split(" → ")
            
            try:
                delay = float(delay_str)
                original_weight = self.graph.edges.get((u, v), 0)
                new_weight = original_weight + delay
                
                # Update the edge weight
                self.graph.update_edge_weight(u, v, new_weight)
                
                msg = (f"Traffic delay of {delay} added to road {u}↔{v}.\n"
                       f"New travel time is {new_weight:.2f}.")
                
                # If there's an active route, automatically recalculate
                if self.path and len(self.path) > 1:
                    msg += "\n\nRecalculating the route..."
                    messagebox.showinfo("Success", msg)
                    self.find_path()  # This will automatically reroute
                else:
                    messagebox.showinfo("Success", msg)
                    self.draw_map()
                
            except ValueError:
                messagebox.showerror("Error", "Invalid delay. Please enter a number.")

    def new_graph(self):
        confirm = messagebox.askyesno("New Graph", "Clear current graph and start new?")
        if confirm:
            self.graph = Graph()
            self.path = []
            self.visited_nodes = []
            self.node_counter = 0
            self.update_node_lists()
            self.draw_map()
            self.result_label.config(text="")
            self.status_label.config(text="New graph created")

    def load_sample(self):
        self.graph = build_sample_graph()
        self.path = []
        self.visited_nodes = []
        self.update_node_lists()
        self.draw_map()
        self.result_label.config(text="")
        self.status_label.config(text="Sample graph loaded")

    def save_graph(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.graph.to_dict(), f, indent=2)
                messagebox.showinfo("Success", "Graph saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save graph: {e}")

    def load_graph(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                self.graph.from_dict(data)
                self.path = []
                self.visited_nodes = []
                self.update_node_lists()
                self.draw_map()
                self.result_label.config(text="")
                messagebox.showinfo("Success", "Graph loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load graph: {e}")


class TrafficDialog(simpledialog.Dialog):
    def __init__(self, parent, title=None, road_options=None):
        self.road_options = road_options
        self.road_var = tk.StringVar()
        self.delay_var = tk.StringVar()
        self.result = None
        super().__init__(parent, title)

    def body(self, master):
        ttk.Label(master, text="Select Road:", font=("Helvetica", 10)).grid(row=0, sticky='w', pady=(5,0))
        self.road_combo = ttk.Combobox(master, textvariable=self.road_var, 
                                       values=self.road_options, state="readonly", width=20)
        self.road_combo.grid(row=1, padx=5, pady=5, sticky='ew')
        self.road_combo.set(self.road_options[0] if self.road_options else "")

        ttk.Label(master, text="Enter Delay (minutes):", font=("Helvetica", 10)).grid(row=2, sticky='w', pady=(10,0))
        self.delay_entry = ttk.Entry(master, textvariable=self.delay_var, width=20)
        self.delay_entry.grid(row=3, padx=5, pady=5, sticky='ew')
        self.delay_entry.insert(0, "0.5")
        
        return self.road_combo

    def apply(self):
        self.result = (self.road_var.get(), self.delay_var.get())


if __name__ == "__main__":
    g = build_sample_graph()
    app = TrafficNavApp(g)
    app.mainloop()