import networkx as nx
import matplotlib.pyplot as plt

# Define graph structure based on the table
edges = [
    ("Start", "Input Fetching"),
    ("Input Fetching", "Input Fetching (1/2)"),
    ("Input Fetching (1/2)", "Input Fetching"),
    ("Input Fetching", "Execution"),
    ("Input Fetching", "Returnal"),
    ("Input Fetching (1/2)", "Execution"),
    ("Input Fetching (1/2)", "Returnal"),
    ("Input Fetching", "Input Fetching (2/2)"),
    ("Input Fetching (2/2)", "Input Fetching"),
    ("Input Fetching", "Execution"),
    ("Input Fetching", "Returnal"),
    ("Input Fetching (2/2)", "Execution"),
    ("Input Fetching (2/2)", "Returnal"),
    ("Input Fetching", "Execution"),
    ("Input Fetching", "Returnal"),
    ("Execution", "Returnal")
]

# Create a directed graph
G = nx.DiGraph()
G.add_edges_from(edges)

# Draw the graph
plt.figure(figsize=(10, 6))
pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # Hierarchical layout
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color="gray", font_size=10)
plt.title("Recursive Calculation Flow")
plt.show()
