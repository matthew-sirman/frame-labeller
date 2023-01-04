import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HGTConv, aggr
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import k_hop_subgraph

import config as cfg


CUDA = torch.cuda.is_available() and cfg.USE_CUDA_IF_AVAILABLE
DEVICE = "cuda" if CUDA else "cpu"


def convert_eds_to_graph(eds_data, predicate2ix, attr2ix):
    """
    Take an EDS object and construct a PyTorch graph.

    Parameters:
    eds_data (EDS): EDS object to create a graph for
    predicate2ix (dict): Lookup for predicate names
    attr2ix (dict): Lookup for attribute names

    Returns:
    graph (Data): PyTorch graph representation
    top_node (int): Index of root node
    root_args (set): Root node argument names and corresponding nodes
    """
    nodes = []
    node2ix = {}

    for node in eds_data.nodes:
        node2ix[node.id] = len(node2ix)
        nodes.append(predicate2ix[node.predicate])

    sources, targets = [], []
    attrs = []
    for node in eds_data.nodes:
        for attr, neighbour in node.edges.items():
            if attr in cfg.IGNORE_ATTRS:
                continue

            sources.append(node2ix[node.id])
            targets.append(node2ix[neighbour])
            attrs.append(attr2ix[attr])

    root_args = set()
    for arg_name, n in eds_data[eds_data.top].edges.items():
        if arg_name in cfg.IGNORE_ATTRS:
            continue
        root_args.add((arg_name, node2ix[n]))

    nodes = torch.tensor(nodes, dtype=torch.long)
    edges = torch.tensor([sources, targets], dtype=torch.long)
    edge_attrs = torch.tensor(attrs, dtype=torch.long)

    graph = Data(x=nodes, edge_index=edges, edge_attr=edge_attrs)
    graph.to(DEVICE)
    top_node = node2ix[eds_data.top]

    return graph, top_node, root_args


class FrameLabeller(nn.Module):
    """
    Frame labeller network.

    Graph neural network model for labelling EDS graphs with their
    primary predicate's frame and roles
    """

    def __init__(self, semlink_dataset,
                 embedding_size=100, heads=1, hidden_node_feature_size=100,
                 output_embedding_size=50,
                 subgraph_hops=3):
        """
        Create a frame labeller model.

        Parameters:
        semlink_dataset (SemLinkDataset): Metadata for index maps
        embedding_size (int): Size of learned embeddings from predicates and
                              attributes
        heads (int): Attention heads for GATConv layer
        hidden_node_feature_size (int): Output size for first GNN layer
        output_embedding_size (int): Output size of final GNN layers
        subgraph_hops (int): Reachable lookahead of argument subgraphs
        """
        super(FrameLabeller, self).__init__()
        self.predicate2ix = semlink_dataset.predicate2ix
        self.attr2ix = semlink_dataset.attr2ix
        self.ix2frame = semlink_dataset.ix2frame
        self.ix2role = semlink_dataset.ix2role
        self.subgraph_hops = subgraph_hops
        self.predicate_embedding = nn.Embedding(len(self.predicate2ix),
                                                embedding_size)
        self.attr_embedding = nn.Embedding(len(self.attr2ix), embedding_size)

        self.gnn = GATv2Conv(embedding_size,
                             hidden_node_feature_size,
                             edge_dim=embedding_size,
                             heads=heads)

        self.arg_gnn = GATv2Conv(hidden_node_feature_size * heads,
                                 hidden_node_feature_size,
                                 edge_dim=embedding_size,
                                 heads=heads)

        self.aggregate_arg = aggr.SoftmaxAggregation()

        self.output_gnn = HGTConv(-1,
                                  output_embedding_size,
                                  (["frame", "role"],
                                   [("frame", "f-loop", "frame"),
                                    ("frame", "edge", "role"),
                                    ("role", "r-loop", "role")]))

        self.frame_predict = nn.Linear(output_embedding_size,
                                       len(self.ix2frame))
        self.role_predict = nn.Linear(output_embedding_size,
                                      len(self.ix2role))

    def forward(self, eds_data, eds_graph=None, eds_root=None, eds_args=None):
        """
        Forward pass through model.

        Parameters:
        eds_data (EDS): EDS graph to predict on
        eds_graph (Data): Precomputed PyTorch data graph
        eds_root (int): Precomputed root node
        eds_args (set): Precomputed arg/attr set

        Returns:
        frame (Tensor): Frame predictions (one-hot)
        args (dict): Argument predictions (one-hot)
        """
        if eds_graph is None:
            eds_data = convert_eds_to_graph(eds_data,
                                            self.predicate2ix,
                                            self.attr2ix)
            eds_graph, eds_root, eds_args = eds_data

        p_embeddings = self.predicate_embedding(eds_graph.x)
        a_embeddings = self.predicate_embedding(eds_graph.edge_attr)

        node_embeddings = self.gnn(x=p_embeddings,
                                   edge_index=eds_graph.edge_index,
                                   edge_attr=a_embeddings)

        intermediate_graph = Data(x=node_embeddings,
                                  edge_index=eds_graph.edge_index,
                                  edge_attr=a_embeddings)

        induced_subgraph = HeteroData()

        induced_subgraph["frame"].x = self.aggregate_arg(node_embeddings)
        induced_subgraph["role"].x = torch.zeros(len(eds_args),
                                                 node_embeddings.size(1))

        induced_edges = torch.zeros(2, len(eds_args), dtype=torch.long)

        arg_lookup = dict(eds_args)

        for i, (arg_name, eds_arg) in enumerate(eds_args):
            node_subset = k_hop_subgraph(eds_arg,
                                         self.subgraph_hops,
                                         eds_graph.edge_index)[0]
            subgraph = intermediate_graph.subgraph(node_subset)
            arg_outputs = self.arg_gnn(x=subgraph.x,
                                       edge_index=subgraph.edge_index,
                                       edge_attr=subgraph.edge_attr)
            induced_subgraph["role"].x[i] = self.aggregate_arg(arg_outputs)
            induced_edges[1, i] = i
            arg_lookup[arg_name] = i

        frame_loop = torch.tensor([[0], [0]]).long()
        arg_index_list = [i for i in range(len(eds_args))]
        arg_loops = torch.tensor([arg_index_list, arg_index_list]).long()
        induced_subgraph["frame", "f-loop", "frame"].edge_index = frame_loop
        induced_subgraph["role", "r-loop", "role"].edge_index = arg_loops
        induced_subgraph["frame", "edge", "role"].edge_index = induced_edges

        output = self.output_gnn(induced_subgraph.x_dict,
                                 induced_subgraph.edge_index_dict)

        frame = F.log_softmax(self.frame_predict(output["frame"]), dim=1)
        roles = F.log_softmax(self.role_predict(output["role"]), dim=1)

        args = {}

        for role_arg in arg_lookup:
            args[role_arg] = roles[arg_lookup[role_arg]]

        return frame[0], args

    def predict(self, eds_data, eds_graph=None, eds_root=None, eds_args=None):
        """
        Predict output classes instead of one-hot prediction tensors.

        Parameters:
        eds_data (EDS): EDS graph to predict on
        eds_graph (Data): Precomputed PyTorch data graph
        eds_root (int): Precomputed root node
        eds_args (set): Precomputed arg/attr set

        Returns:
        frame (str): Frame prediction
        args (dict): Arg predictions
        """
        frame, args = self.forward(eds_data, eds_graph, eds_root, eds_args)
        frame = self.ix2frame[torch.argmax(frame).item()]
        args = {arg: self.ix2role[torch.argmax(args[arg]).item()] for arg in args}

        return frame, args
