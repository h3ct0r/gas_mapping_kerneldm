import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import networkx as nx

filename = '/tmp/GMRF_v2.txt'
delimiter = ' '  # import a file with space delimiters
data = []

# quoting=csv.QUOTE_NONNUMERIC to automatically converto to float
for row in csv.reader(open(filename), delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC):
    data.append(row)

connection_size = len(data)
print("connection_size:", connection_size)

x = []
y = []

for i in range(connection_size):
    a = (data[i][0], data[i][1])
    b = (data[i][2], data[i][3])

    x.append(a[0])
    x.append(b[0])
    y.append(a[1])
    y.append(b[1])

max_x = int(max(x) + 1)
max_y = int(max(y) + 1)

print("max_x:", max_x, "max_y:", max_y)

edge_list = []
pos = {}

for i in range(connection_size):
    a = (data[i][0], data[i][1])
    b = (data[i][2], data[i][3])

    id_a = int((a[1] * max_y) + a[0])
    id_b = int((b[1] * max_y) + b[0])

    pos[id_a] = a
    pos[id_b] = b

    edge_list.append([id_a, id_b])
    edge_list.append([id_b, id_a])

G = nx.Graph()
G.add_edges_from(edge_list)
nx.draw(G, pos, edge_labels=False, node_size=1.8)
plt.show()

# p = (15, 17)
# k_p = int((p[1] * max_y) + p[0])

# path = nx.algorithms.bfs_edges(G, source=k_p, depth_limit=10)
# bfs_path = list(path)
# xx = []
# yy = []

# for e in bfs_path:
#     p1 = pos[e[0]]
#     p2 = pos[e[1]]

#     # print p1, p2
#     xx.append(p1[0])
#     xx.append(p2[0])
#     yy.append(p1[1])
#     yy.append(p2[1])

# plt.scatter(xx, yy)
# plt.scatter([p[0]], [p[1]], c='red')
# plt.show()