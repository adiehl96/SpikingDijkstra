from random import randint
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def get_examplegraph():
    wg = [(0, 1, 4), (0, 7, 7), (1, 2, 8), (1, 7, 11)]
    nxwg = {}
    for s, e, w in wg:
        if s in nxwg:
            nxwg[s][e] = {"weight": w}
        else:
            nxwg[s] = {e: {"weight": w}}
    nxwg = nx.from_dict_of_dicts(nxwg)
    nxwg = nxwg.to_directed()
    return nxwg


def draw_examplegraph():
    g = get_examplegraph()
    pos = nx.spring_layout(g)
    nx.draw(g, pos=pos, with_labels=True)
    elabels = {(s, t): g[s][t]["weight"] for (s, t) in g.edges}
    nx.draw_networkx_edge_labels(g, pos, elabels)
    plt.draw()
    pass


def is_equal(g, pathdict1, pathdict2):
    """
    Checks equality of two dictionaries containing paths through graph g.
    g: networkx graph
    pathdict1: python dict e.g. {0: [0], 1: [0, 1], 2: [0, 2], 3: [0, 3]}
    pathdict2: python dict e.g. {0: [0], 1: [0, 1], 2: [0, 2], 3: [0, 3]}
    return: Boolean truth value
    """

    # First check if both dicts contain the same elements to prevent retrieval errors
    for n in pathdict1 | pathdict2:
        if (n not in pathdict1) or (n not in pathdict2):
            return False

    # Then check if we can establish that for each path both solutions visit the same nodes or cover the same distance.
    for n in pathdict1:
        if pathdict1[n] == pathdict2[n]:
            continue
        elif nx.path_weight(g, pathdict1[n], weight="weight") == nx.path_weight(
            g, pathdict2[n], weight="weight"
        ):
            continue
        else:
            return False
    return True


def get_test_graphs(N, P, k):
    N = np.atleast_1d(N)
    P = np.atleast_1d(P)
    test_graphs = []
    s = 0  # seed to get the same random graph for testing
    for n in N:  # number of nodes
        for p in P:  # probability of edge creation
            for _ in range(0, k):
                source = randint(0, n - 1)
                s += 1
                g = nx.fast_gnp_random_graph(n, p, s)
                while not nx.is_connected(g):
                    s += 1
                    g = nx.fast_gnp_random_graph(n, p, s)

                # ATTENTION on randint below: min 1 as we need to be able to represent this number with delay and max 12 as we don't want to have to run the simulation for very long
                for e in g.edges:
                    g[e[0]][e[1]]["weight"] = randint(1, 12)
                weights = {(s, t): g[s][t]["weight"] for (s, t) in g.edges}
                gd = {"n": n, "p": p, "s": s, "source": source, "weights": weights}
                test_graphs.append(gd)
    return test_graphs
