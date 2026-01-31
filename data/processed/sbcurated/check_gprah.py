import networkx as nx
from collections import Counter

graph_path = "./cfg_cg_compressed_graphs.gpickle"
G = nx.read_gpickle(graph_path)


prefixes = set()
for node, data in G.nodes(data=True):
    print(data)
    source_file = data.get("source_file")
    if source_file:
        prefix = source_file.split('/')[0]
        prefixes.add(prefix)

print("Các prefix khác nhau của source_file là:")
for p in prefixes:
    print("-", p)



