import random

import delphin.codecs.eds as eds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HGTConv, aggr

import config as cfg
from loader import convert_eds_to_hetero_graph


CUDA = torch.cuda.is_available() and cfg.USE_CUDA_IF_AVAILABLE
DEVICE = "cuda" if CUDA else "cpu"


class FrameLabeller(nn.Module):
    """
    Frame labeller network.

    Graph neural network model for labelling EDS graphs with their
    primary predicate's frame and roles
    """

    def __init__(self, semlink_dataset,
                 embedding_size=100, heads=1, hidden_node_feature_size=100):
        """
        Create a frame labeller model.

        Parameters:
        semlink_dataset (SemLinkDataset): Metadata for index maps
        embedding_size (int): Size of learned embeddings from predicates and
                              attributes
        heads (int): Attention heads for GATConv layer
        hidden_node_feature_size (int): Output size for first GNN layer
        output_embedding_size (int): Output size of final GNN layers
        """
        super(FrameLabeller, self).__init__()
        self.predicate2ix = semlink_dataset.predicate2ix
        self.attr2ix = semlink_dataset.attr2ix
        self.ix2frame = semlink_dataset.ix2frame
        self.ix2role = semlink_dataset.ix2role
        self.hidden_size = hidden_node_feature_size

        self.predicate_embedding = nn.Embedding(len(self.predicate2ix), embedding_size)
        self.attr_embedding = nn.Embedding(len(self.attr2ix), embedding_size)

        self.gnn = HGTConv(embedding_size, hidden_node_feature_size,
                           (["node", "edge"], [("node", "true-edge", "node"),
                                               ("node", "edge-in", "edge"),
                                               ("edge", "edge-out", "node")]),
                           heads=heads)

        self.aggregate_graph = aggr.SoftmaxAggregation()

        self.root_node_predict = GATv2Conv(hidden_node_feature_size, 1, edge_dim=hidden_node_feature_size)
        self.frame_predict = nn.Linear(hidden_node_feature_size * 2, len(self.ix2frame))
        self.role_predict = nn.Linear(hidden_node_feature_size * 2, len(self.ix2role))

    def forward(self, eds_graph, teacher_target=None, teacher_force_ratio=0.5):
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
        if isinstance(eds_graph, eds.EDS):
            eds_graph, ix2node = convert_eds_to_hetero_graph(eds_graph, self.predicate2ix, self.attr2ix)

        p_embeddings = self.predicate_embedding(eds_graph["node"].x)
        a_embeddings = self.predicate_embedding(eds_graph["edge"].x)

        hidden = self.gnn({"node": p_embeddings, "edge": a_embeddings}, eds_graph.edge_index_dict)
        root_hidden_nodes = hidden["node"]
        frame_hidden_nodes = hidden["node"]
        role_hidden_nodes = hidden["node"]
        root_hidden_edges = hidden["edge"]
        frame_hidden_edges = hidden["edge"]
        role_hidden_edges = hidden["edge"]

        root_preds = self.root_node_predict(x=root_hidden_nodes,
                                            edge_index=eds_graph["node", "true-edge", "node"].edge_index,
                                            edge_attr=root_hidden_edges)
        root_preds = F.log_softmax(root_preds, dim=0).squeeze(1)

        graph_node_rep = self.aggregate_graph(frame_hidden_nodes)
        graph_edge_rep = self.aggregate_graph(frame_hidden_edges)
        if graph_edge_rep.size(0) == 0:
            graph_edge_rep = torch.zeros_like(graph_node_rep)

        graph_rep = torch.cat((graph_node_rep, graph_edge_rep), 1)

        frame_preds = F.log_softmax(self.frame_predict(graph_rep), dim=1).squeeze(0)

        graph_role_rep = self.aggregate_graph(role_hidden_nodes).repeat(role_hidden_edges.size(0), 1)
        role_preds = self.role_predict(torch.cat((role_hidden_edges, graph_role_rep), dim=1))

        if teacher_target is not None and random.random() < teacher_force_ratio:
            non_adjacent_edges = torch.argwhere(eds_graph["edge-in"].edge_index[0] != teacher_target)
        else:
            non_adjacent_edges = torch.argwhere(eds_graph["edge-in"].edge_index[0] != torch.argmax(root_preds))

        role_preds[non_adjacent_edges] = 0
        role_preds = F.log_softmax(role_preds, dim=1)

        return (root_preds, frame_preds), role_preds

    def predict(self, eds_graph, ix2node=None):
        """
        Predict output classes instead of one-hot prediction tensors.

        Parameters:
        eds_graph (Data or EDS): EDS graph to predict on
        ix2node (dict): Mapping from graph indices to nodes

        Returns:
        root (str): Graph node to annotate the frame onto
        frame (str): Frame to annotate
        args (dict): Predicted roles
        """
        if isinstance(eds_graph, eds.EDS):
            eds_graph, ix2node = convert_eds_to_hetero_graph(eds_graph, self.predicate2ix, self.attr2ix)

        (root_preds, frame_preds), role_preds = self.forward(eds_graph)
        root_preds = torch.argmax(root_preds, dim=0)
        frame_preds = torch.argmax(frame_preds, dim=0)
        role_preds = torch.argmax(role_preds, dim=1)

        root = ix2node[root_preds.item()]
        frame = self.ix2frame[frame_preds.item()]

        roles = []
        for i, edge in enumerate(role_preds):
            if edge.item() != 0:
                s, t = eds_graph["node", "true-edge", "node"].edge_index[:, i]
                roles.append((ix2node[s.item()], ix2node[t.item()], self.ix2role[edge.item()]))

        return (root, frame), roles
