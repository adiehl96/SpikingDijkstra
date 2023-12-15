from pprint import pprint
import numpy as np
import networkx as nx

from simsnn.core.networks import Network
from simsnn.core.simulators import Simulator


def spl(g, source=0, target=2):
    """
    Shortest Path Length From Source to Target (SPL)
    The following code returns the length of the shortest path from node 0 to node 2 in the examplegraph.
    """

    # Create bidirectional edges, as we need for this algorithm to work.
    g = g.to_directed()
    net = Network()
    sim = Simulator(net)

    N = {n: net.createLIF(ID=f"{n}", V_init=999, thr=1000) for n in g}
    __S__ = {
        e: net.createSynapse(
            ID=f"{e}", pre=N[e[0]], post=N[e[1]], d=g[e[0]][e[1]]["weight"]
        )
        for e in g.edges
    }

    # Let the source neuron spike immediately, to start the path length calculation
    N[source].V = 1000

    sim.raster.addTarget(N[target])
    sim.run(steps=15, plotting=False)

    rasterdata = sim.raster.get_measurements().T.flatten()
    snnpathlength = np.where(rasterdata > 0)[0].item()

    nxpathlength = nx.shortest_path_length(g, source, target, weight="weight")
    assert nxpathlength == snnpathlength

    return snnpathlength


def sp(g, source=0, target=2):
    """
    Shortest Path From Source to Target (SP)
    The following code returns the shortest path from node 0 to node 2 in the examplegraph.
    """

    # Networkx Solution
    nxpath = nx.shortest_path(g, source, target, weight="weight")

    # simsnn solution
    net = Network()
    sim = Simulator(net)

    N = {n: net.createLIF(ID=f"{n}", V_init=999, thr=1000) for n in g}
    S = {
        e: net.createSynapse(
            ID=f"{e}", pre=N[e[0]], post=N[e[1]], d=g[e[0]][e[1]]["weight"]
        )
        for e in g.edges
    }

    D = {f"{p}_d_{n}": net.createLIF(ID=f"{p}_d_{n}") for p, n in g.edges}
    __N_D__ = {
        (p, n): net.createSynapse(
            ID=f"{(p,n)}", pre=N[p], post=D[f"{p}_d_{n}"], d=g[p][n]["weight"]
        )
        for p, n in g.edges
    }

    E = {f"{p}_e_{n}": net.createLIF(ID=f"{p}_e_{n}", thr=2, m=0) for p, n in g.edges}
    __D_E__ = {
        (f"{p}_d_{n}", f"{p}_e_{n}"): net.createSynapse(
            ID=f"({p}_d_{n}, {p}_e_{n})", pre=D[f"{p}_d_{n}"], post=E[f"{p}_e_{n}"]
        )
        for (p, _, _, _, n) in E
    }
    __N_E__ = {
        (int(n), f"{p}_e_{n}"): net.createSynapse(
            ID=f"(n, {p}_e_{n})", pre=N[int(n)], post=E[f"{p}_e_{n}"]
        )
        for (p, _, _, _, n) in E
    }

    N[
        source
    ].V = 1000  # Let the source neuron spike immediately, to start the path length calculation

    order = [e for e in E]
    sim.raster.addTarget([E[e] for e in order])
    sim.run(steps=50, plotting=False)

    rasterdata = sim.raster.get_measurements().T
    tts = np.where(rasterdata > 0)

    path = lambda h, t, d: t if t[0] == h else path(h, [d.get(t[0], h)] + t, d)

    L = {int(order[k][-1]): int(order[k][0]) for (k, v) in zip(tts[0], tts[1])}
    snnpath = path(source, [target], L)

    assert nxpath == snnpath
    # print("Succes")
    return snnpath


def ssspl(g, source):
    """
    Single-Source Shortest Path Length to All Nodes (SSSPL)
    The following code returns the lengths of the shortest paths from node 0 to all nodes in the exampleg.
    """
    # Create bidirectional edges, as we need for this algorithm to work.
    g = g.to_directed()
    net = Network()
    sim = Simulator(net)

    N = {n: net.createLIF(ID=f"{n}", V_init=999, thr=1000) for n in g}
    __S__ = {
        e: net.createSynapse(
            ID=f"{e}", pre=N[e[0]], post=N[e[1]], d=g[e[0]][e[1]]["weight"]
        )
        for e in g.edges
    }

    pn = net.createInputTrain(ID="pn", train=[1], loop=False)
    __pn_src__ = net.createSynapse(ID=f"{pn.ID}-{source}", pre=pn, post=N[source])

    order = [n for n in N]
    sim.raster.addTarget([N[n] for n in order] + [pn])
    sim.multimeter.addTarget([N[n] for n in order] + [pn])
    sim.run(steps=50, plotting=False)

    rasterdata = sim.raster.get_measurements().T

    snnpathlengths = np.where(rasterdata > 0)
    snnpathlengths = {n: l - 1 for (n, l) in zip(order, snnpathlengths[1])}
    nxpathlengths = nx.single_source_dijkstra_path_length(g, source, weight="weight")
    assert nxpathlengths == snnpathlengths

    return snnpathlengths


def sssp(g, source):
    """
    Single-Source Shortest Path to All Nodes (SSSP)
    """
    # Create bidirectional edges, as we need for this algorithm to work.
    g = g.to_directed()
    net = Network()
    sim = Simulator(net)

    N = {n: net.createLIF(ID=f"{n}", V_init=999, thr=1000) for n in g}
    __S__ = {
        e: net.createSynapse(
            ID=f"{e}", pre=N[e[0]], post=N[e[1]], d=g[e[0]][e[1]]["weight"]
        )
        for e in g.edges
    }

    D = {f"{p}_d_{n}": net.createLIF(ID=f"{p}_d_{n}") for p, n in g.edges}
    __N_D__ = {
        (p, n): net.createSynapse(
            ID=f"{(p,n)}", pre=N[p], post=D[f"{p}_d_{n}"], d=g[p][n]["weight"]
        )
        for p, n in g.edges
    }

    E = {f"{p}_e_{n}": net.createLIF(ID=f"{p}_e_{n}", thr=2, m=0) for p, n in g.edges}
    pprint(E)
    __D_E__ = {}
    for e in E:
        p = e.split("_")[0]
        n = e.split("_")[-1]
        synapse = net.createSynapse(
            ID=f"({p}_d_{n}, {e})", pre=D[f"{p}_d_{n}"], post=E[e]
        )
        __D_E__[(f"{p}_d_{n}", e)] = synapse
    # __D_E__ = {
    #     (f"{p}_d_{n}", f"{p}_e_{n}"): net.createSynapse(
    #         ID=f"({p}_d_{n}, {p}_e_{n})", pre=D[f"{p}_d_{n}"], post=E[f"{p}_e_{n}"]
    #     )
    #     for (p, _, _, _, n) in E
    # }
    __N_E__ = {}
    for e in E:
        # p = e.split("_")[0]
        n = e.split("_")[-1]
        synapse = net.createSynapse(ID=f"({n}, {e})", pre=N[int(n)], post=E[e])
        __N_E__[(int(n), e)] = synapse

    # __N_E__ = {
    #     (int(n), f"{p}_e_{n}"): net.createSynapse(
    #         ID=f"(n, {p}_e_{n})", pre=N[int(n)], post=E[f"{p}_e_{n}"]
    #     )
    #     for (p, _, _, _, n) in E
    # }

    N[
        source
    ].V = 1000  # Let the source neuron spike immediately, to start the path length calculation

    order = np.array([e for e in E])
    sim.raster.addTarget([E[e] for e in order])
    sim.run(steps=50, plotting=False)

    rasterdata = sim.raster.get_measurements().T
    spiked = order[(rasterdata > 0).any(axis=1)]

    L = {}
    for s in spiked:
        p = s.split("_")[0]
        n = s.split("_")[-1]
        L[int(n)] = int(p)
    # L = {int(n): int(p) for p, _, _, _, n in spiked}
    path = lambda h, t, d: t if t[0] == h else path(h, [d.get(t[0], h)] + t, d)
    return {n: path(source, [n], L) for n in L} | {source: [source]}
