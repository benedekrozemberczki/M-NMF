"""Data reader utils."""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from texttable import Texttable

def modularity_generator(G):
    """
    Function to generate a modularity matrix.
    :param G: Graph object.
    :return laps: Modularity matrix.
    """
    print("Modularity calculation.\n")
    degs = nx.degree(G)
    e_count = len(nx.edges(G))
    modu = np.array([[float(degs[n_1]*degs[n_2])/(2*e_count) for n_1 in nx.nodes(G)] for n_2 in tqdm(nx.nodes(G))], dtype=np.float64)
    return modu

def overlap_generator(G):
    """
    Function to generate a neighbourhood overlap matrix (second-order proximity matrix).
    :param G: Graph object.
    :return laps: Overlap matrix.
    """
    print("Second order proximity calculation.\n")
    degs = nx.degree(G)
    sets = {node:set(G.neighbors(node)) for node in nx.nodes(G)}
    laps = np.array([[float(len(sets[n_1].intersection(sets[n_2])))/(float(degs[n_1]*degs[n_2])**0.5) if n_1 != n_2 else 0.0 for n_1 in nx.nodes(G)] for n_2 in tqdm(nx.nodes(G))], dtype=np.float64)
    return laps

def graph_reader(input_path):
    """
    Function to read a csv edge list and transform it to a networkx graph object.
    :param input_path: Path to the edge list csv.
    :return graph: NetworkX grapg object.
    """
    edges = pd.read_csv(input_path)
    graph = nx.from_edgelist(edges.values.tolist())
    return graph

def log_setup(args_in):
    """
    Function to setup the logging hash table.
    :param args_in: Arguments used for the model.
    :return log: The updated log.
    """
    log = dict()
    log["times"] = []
    log["cluster_quality"] = []
    log["params"] = vars(args_in)
    return log

def json_dumper(data, path):
    """
    Function to dump the logs and assignments.
    :param data: The dictionary to store.
    :param path: Path where the dictionary is saved.
    """
    with open(path, 'w') as outfile:
        json.dump(data, outfile)

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), v] for k, v in args.items()])
    print(t.draw())

def log_updater(log, repetition, optimization_time, modularity_score):
    """
    Function to update the log object.
    :param log: The log to update.
    :param repetition: Number of iteration.
    :param optimization_time: Time needed for optimization.
    :param modularity_score: Modularity after the optimization round.
    :return log: Updated log.
    """
    index = repetition + 1
    log["times"] = log["times"] + [[index, optimization_time]]
    log["cluster_quality"] = log["cluster_quality"] + [[index, modularity_score]]
    return log

def loss_printer(log):
    """
    Function to print the logs in a nice tabular format.
    :param log: Dictionary with the log.
    """
    t = Texttable()
    t.add_rows([["Round", "Modularity"]])
    t.add_rows([k for k in log["cluster_quality"]])
    print(t.draw())
