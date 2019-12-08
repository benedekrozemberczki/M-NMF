"""Running M-NMF."""

from modularity_nmf import MNMF
from param_parser import parameter_parser
from calculation_helper import tab_printer

def create_and_run_model(args):
    """
    Function to read the graph, create an embedding and train it.
    :param args: Object with parameters - paths and parameters.
    """
    tab_printer(args)
    model = MNMF(args)
    model.optimize()

if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)
