import sys
from random import randrange, seed

seed(0)

a, b, e = [int(x) for x in sys.argv[1:]]

# ugly way to ensure that there are no disconnected nodes
while True:
    edges = set()
    for i in range(a):
        edges.add((i, randrange(b)))
    b_nodes_done = set([j for (i, j) in edges])
    for j in range(b):
        if j not in b_nodes_done:
            edges.add((randrange(a), j))
    while len(edges) < e:
        edges.add((randrange(a), randrange(b)))
    if len(edges) == e:
        break

with open(f"random_{a}_{b}_{e}.gr", "w") as f:
    print("p ocr", a, b, e, file=f)
    for x, y in edges:
        print(x + 1, a + y + 1, file=f)
