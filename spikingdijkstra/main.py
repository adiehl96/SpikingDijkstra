from random import randint

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from algorithms import ssspl, sssp
from graphsupport import get_test_graphs, is_equal


def test_ssspl():
    gds = get_test_graphs(range(3, 12), np.linspace(0.5, 1, 2), 5)
    for gd in gds:
        g = nx.fast_gnp_random_graph(gd["n"], gd["p"], seed=gd["s"])
        for e0, e1 in gd["weights"]:
            g[e0][e1]["weight"] = gd["weights"][(e0, e1)]
        nxpathlengths = nx.single_source_dijkstra_path_length(g, gd["source"])
        snnpathlengths = ssspl(g, gd["source"])
        try:
            assert nxpathlengths == snnpathlengths
        except AssertionError:
            print("Failure, nxpathlengths != snnpathlengths for graph:", gd)
    return nxpathlengths == snnpathlengths


def test_sssp():
    gds = get_test_graphs(range(3, 12), np.linspace(0.5, 1, 2), 5)
    for gd in gds:
        g = nx.fast_gnp_random_graph(gd["n"], gd["p"], seed=gd["s"])
        for e0, e1 in gd["weights"]:
            g[e0][e1]["weight"] = gd["weights"][(e0, e1)]
        nxpaths = nx.single_source_dijkstra(g, gd["source"])[1]
        snnpaths = sssp(g, gd["source"])
        print(f"checking if snnpaths {snnpaths} equals nxpaths {nxpaths}")
        try:
            assert is_equal(g, nxpaths, snnpaths)
        except AssertionError:
            print("Failure, nxpaths != snnpaths for graph:", gd)
            pos = nx.spring_layout(g)
            nx.draw(g, pos=pos, with_labels=True)
            nx.draw_networkx_edge_labels(g, pos, gd["weights"])
            plt.draw()
            plt.show()
    return is_equal(g, nxpaths, snnpaths)


test_ssspl()
test_sssp()
