"""
Data loader module.

Handles loading of EDS and SemLink databases
"""
import os
import delphin.codecs.eds as eds
import torch
import random
import json
from intervaltree import IntervalTree

from torch_geometric.data import HeteroData
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize

import config as cfg


if cfg.RANDOM_SEED is not None:
    random.seed(cfg.RANDOM_SEED)


def lexer(s):
    """
    Lexer for s-expression parser.

    Parameters:
    s (str): Raw source string expression

    Returns:
    list: derived tokens
    """
    chars = list(s)
    tokens = []

    tok = ""

    while chars:
        c = chars.pop(0)
        if c in {'(', ')'}:
            if tok != "":
                tokens.append(tok)
                tok = ""
            tokens.append(c)
        elif c.isspace():
            if tok != "":
                tokens.append(tok)
                tok = ""
        else:
            tok += c

    if tok != "":
        tokens.append(tok)

    return tokens


def sexp_parser(s):
    """
    Parser for s-expressions.

    Given a string, produce an s-expression as a list of lists.

    Parameters:
    s (str): Raw source string expression

    Returns:
    list: list representation of s-expression
    """
    tokens = lexer(s)

    def parse_atom():
        tok = tokens.pop(0)
        if tok == '(':
            return parse_tail()
        if tok == ')':
            return None
        return tok

    def parse_tail():
        ls = []
        while (item := parse_atom()) is not None:
            ls.append(item)
        return ls

    exps = []

    while tokens:
        exps.append(parse_atom())

    return exps


def parse_penn_treebank(ptb_sources_dir, save_as_json=None):
    """
    Walk the Penn TreeBank directory and load and parse each file.

    Parameters:
    ptb_sources_dir (str): Path to Penn TreeBank directory
    save_as_json (str, optional): Path to save parsed data in JSON format

    Returns:
    dict: A mapping from sentences to the file and index they appear at
    """
    def replace_special_branches(sexps):
        if len(sexps) == 2:
            if sexps[0] == "-NONE-":
                return ["-NONE-", None]
            elif sexps[0] == "-LRB-":
                return ["LRB", "("]
            elif sexps[0] == "-RRB-":
                return ['RRB', ")"]
        branches = []
        for branch in sexps:
            if isinstance(branch, list):
                branches.append(replace_special_branches(branch))
            else:
                branches.append(branch)
        return branches

    def extract_words(sexps):
        sexps = replace_special_branches(sexps)

        def recurse_branch(branch):
            if isinstance(branch, list):
                last = branch[-1]
                if not isinstance(last, list):
                    return [last]
                else:
                    return [recurse_branch(b) for b in branch]
            else:
                return []

        def flatten(ls):
            if isinstance(ls, list):
                out = []
                for item in ls:
                    out += flatten(item)
                return out
            else:
                return [ls]

        def postprocess(s):
            sent = []
            for i, w in enumerate(flatten(s)):
                if w is not None:
                    sent.append((i, w))
            return sent

        return [postprocess(recurse_branch(s)) for s in sexps]

    file_data = {}
    for root, dirs, files in os.walk(ptb_sources_dir):
        for f_name in files:
            if f_name.endswith(".parse"):
                with open(os.path.join(root, f_name), "r") as in_f:
                    file_data[f_name] = in_f.read().replace("’", "'")

    parsed_files = {}
    for i, f_name in enumerate(file_data):
        parsed_files[f_name] = sexp_parser(file_data[f_name])

    sentence_to_ptb_lookup = {}

    for wsj in parsed_files:
        for i, sentence in enumerate(extract_words(parsed_files[wsj])):
            s_string = " ".join([w for _, w in sentence])
            sentence_to_ptb_lookup[s_string] = (wsj, i, sentence)

    if save_as_json is not None:
        json_data = json.dumps(sentence_to_ptb_lookup)
        with open(save_as_json, "w") as out_file:
            out_file.write(json_data)

    return sentence_to_ptb_lookup


def load_penn_treebank(load_path):
    """
    Load precomputed Penn TreeBank parsed data in JSON format.

    This is helpful as the s-expression parser is quite slow.

    Parameters:
    load_path (str): Path to JSON file

    Returns:
    dict: A mapping from sentences to the file and index they appear at
    """
    with open(load_path, "r") as in_file:
        sentence_to_ptb_lookup = json.load(in_file)

    return sentence_to_ptb_lookup


def load_eds_data_files(eds_sources):
    """
    Load EDS database.

    Parameters:
    eds_sources (str): Path to EDS source database

    Returns:
    list: List of pairs of file name and contents
    """
    files = []
    for journal in os.listdir(eds_sources):
        if not os.path.isdir(os.path.join(eds_sources, journal)):
            continue
        for f in os.listdir(os.path.join(eds_sources, journal)):
            if f[0] == ".":
                continue
            with open(os.path.join(eds_sources, journal, f)) as in_f:
                files.append((f, in_f.read()))

    return files


def extract_relevant_from_eds_document(doc):
    """
    Given a document, find and extract the EDS section.

    Parameters:
    doc (str): The raw document text

    Returns:
    raw_sentence (str): The sentence data from the EDS document
    token_positions (list): Pairs of tokens from the sentence and their
                            start pointer
    eds (str): The EDS section of the document
    """
    token_pos_region = False
    token_positions = []

    eds_region = False
    eds = []

    lines = doc.splitlines()

    raw_sentence_line = lines[5]
    raw_sentence_start = raw_sentence_line.index("`")
    raw_sentence = raw_sentence_line[raw_sentence_start + 1:-1]
    raw_sentence = raw_sentence.replace("-", " - ")
    raw_sentence = " ".join(word_tokenize(raw_sentence))

    for line in lines:
        if line == "":
            continue
        if line[0] == "<" and token_pos_region is not None:
            token_pos_region = True
        elif line[0] == ">":
            token_pos_region = None
        elif token_pos_region:
            line_elements = line.strip(" ()").split(", ")
            token_start = int(line_elements[3].strip("<>").split(":")[0])
            word = line_elements[5]
            word = word.strip("\"")
            word = word.replace("’", "'").replace("“", "``").replace("”", "''")
            word = word.replace("–", "--").replace("‘", "`").replace("…", "...")
            token_positions.append((word, token_start))

        if line[0] == "{":
            eds_region = True
        if eds_region:
            eds.append(line)
        if line[0] == "}":
            break

    eds = "\n".join(eds)

    return raw_sentence, token_positions, eds


def parse_arg(arg_string):
    """
    Parse SemLink frame argument.

    These are formed as:
    [(ArgPointer)-ARGX=(VN Role);(FN Role/Optional Extra Fn Roles)]

    Parameters:
    arg_string (str): Raw unprocessed argument string

    Returns:
    loc (str): Location information for argument (ArgPointer)
    arg (str): Argument name (ARGX)
    fn_role (str): Frame net role (FN Role)
    """
    loc, arg_string = arg_string.split("-", 1)
    loc = loc.split("*")[0]
    loc = loc.split(":")
    cfrom = int(loc[0])
    cto = int(loc[-1])
    loc = cfrom, cfrom + cto + 1
    if "=" not in arg_string:
        return loc, arg_string

    arg, arg_string = arg_string.split("=", 1)

    if ";" not in arg_string:
        return loc, arg

    _, fn_role = arg_string.split(";", 1)
    fn_role = fn_role.split("/")[0]
    return loc, arg, fn_role


def load_semlink(semlink_path):
    """
    Load the SemLink data file.

    Parameters:
    semlink_path (str): File path to SemLink data file

    Returns:
    semlinks (list): Parsed SemLink entries
    semlink_lookup (dict): Mapping from sentences and token indices to
                           sentence data
    """
    semlinks = []
    semlink_lookup = {}
    with open(semlink_path) as semlink:
        for entry in semlink:
            entry = entry.strip().split(maxsplit=10)
            doc = entry[0]
            sentence = int(entry[1])
            token = int(entry[2])
            # Entries 3, 4 and 5 are ignored
            frame = entry[6]
            if frame in {"IN", "NF"}:
                continue
            # Entries 7, 8 and 9 are ignored
            args = entry[10]

            args = [parse_arg(arg_string) for arg_string in args.split()]

            semlinks.append((doc, sentence, token, frame, args))

            key = os.path.basename(doc), sentence
            if key in semlink_lookup:
                semlink_lookup[key][token] = frame, args
            else:
                semlink_lookup[key] = {token: (frame, args)}

    return semlinks, semlink_lookup


def create_predicate_and_attribute_indices(eds_dataset):
    """
    Create mappings from predicates and attributes to integers.

    Parameters:
    eds_dataset (list): Dataset of EDS graphs

    Returns:
    predicate2ix (dict): Mappings from predicate names to indices
    attr2ix (dict): Mappings from attribute names to indices
    """
    predicate2ix = {}
    attr2ix = {}

    for eds_data in eds_dataset:
        for node in eds_data.nodes:
            if node.predicate not in predicate2ix:
                predicate2ix[node.predicate] = len(predicate2ix)
            for attr in node.edges:
                if attr not in attr2ix:
                    attr2ix[attr] = len(attr2ix)

    return predicate2ix, attr2ix


def create_frame_and_role_indices(semlink_dataset):
    """
    Create bijective mappings for frames and roles to integers.

    These are needed for use in neural networks

    Parameters:
    semlink_dataset (list): The parsed dataset of SemLinks

    Returns:
    frame2ix (dict): Mappings from frame names to indices
    ix2frame (dict): Mappings from indices to frame names
    role2ix (dict): Mappings from role names to indices
    ix2role (dict): Mappings from indices to role names
    """
    frame2ix = {"<NONE>": 0}
    ix2frame = {0: "<NONE>"}
    role2ix = {"<NONE>": 0}
    ix2role = {0: "<NONE>"}

    for _, _, _, frame, args in semlink_dataset:
        if frame not in frame2ix:
            frame_id = len(frame2ix)
            frame2ix[frame] = frame_id
            ix2frame[frame_id] = frame
        for arg in args:
            if len(arg) == 3:
                role = arg[2]
                if role not in role2ix:
                    role_id = len(role2ix)
                    role2ix[role] = role_id
                    ix2role[role_id] = role

    return frame2ix, ix2frame, role2ix, ix2role


def connect_eds_to_semlink(eds_dataset, deepbank_ptb_lookup, semlink_lookup):
    """
    Pair entries in the SemLink database to EDS graphs.

    Parameters:
    eds_dataset (list): Entries from the EDS dataset
    semlink_dataset (list): Entries from the SemLink dataset

    Returns:
    list: Tuples annotating EDS graphs with their frames and roles
    """
    # Remove spaces to improve hit rate
    sentence_to_ptb = {}
    for k, v in deepbank_ptb_lookup.items():
        sentence_to_ptb[k.replace(" ", "")] = v

    annotated_eds = []

    for i, (eds_file_name, entry) in enumerate(eds_dataset):
        raw_sentence, token_positions, eds_data = extract_relevant_from_eds_document(entry)

        key = raw_sentence.replace(" ", "")
        if key not in sentence_to_ptb:
            continue

        wsj, sentence_id, ix2token = sentence_to_ptb[key]
        ix2token = dict(ix2token)

        if (wsj, sentence_id) not in semlink_lookup:
            continue

        annotations = semlink_lookup[wsj, sentence_id]

        tokens_to_positions = IntervalTree()
        try:
            for ix, ptb_word in ix2token.items():
                db_word, pos = token_positions[0]
                db_word.index(ptb_word)

                db_word = db_word.removeprefix(ptb_word).strip()
                tokens_to_positions[ix:ix+1] = pos

                if db_word == "":
                    token_positions.pop(0)
                else:
                    token_positions[0] = db_word, pos
        except ValueError:
            continue

        try:
            eds_data = eds.decode(eds_data)
        except eds.EDSSyntaxError:
            continue

        node_intervals = IntervalTree()
        labelled_nodes = []
        for node in eds_data.nodes:
            node_intervals[node.cfrom:node.cto] = node
        for token in annotations:
            frame, args = annotations[token]
            pos = tokens_to_positions[token]
            pos = list(pos)[0].data
            matching_intervals = node_intervals[pos]
            if len(matching_intervals) == 0:
                continue
            node = min(matching_intervals, key=lambda i: i.length()).data
            labelled_nodes.append((node, frame, args))

        root_node = eds_data[eds_data.top]

        def distance_from_root(node):
            to_check = [root_node]
            depth = 0
            at_depth = 1
            next_depth = 0
            visited = set()
            while to_check:
                check = to_check.pop(0)
                visited.add(check.id)
                at_depth -= 1
                if check.id == node.id:
                    return depth
                for neighbour in check.edges.values():
                    if neighbour in visited:
                        continue
                    to_check.append(eds_data[neighbour])
                    next_depth += 1
                if at_depth == 0:
                    at_depth = next_depth
                    next_depth = 0
                    depth += 1
            return float("inf")

        if len(labelled_nodes) == 1:
            root_predicate, frame, args = labelled_nodes[0]
        else:
            def key(node_data):
                return distance_from_root(node_data[0])
            root_predicate, frame, args = min(labelled_nodes, key=key)

        # matched_frames = {}
        matched_eds_annotated_edges = {}

        matched_frames = (node.id, frame)
        eds_primary_edges = IntervalTree()
        for edge_label, neighbour in node.edges.items():
            neighbour = eds_data[neighbour]
            eds_primary_edges[neighbour.cfrom:neighbour.cto] = neighbour

        roles_exist_in_semlink = False

        for arg in args:
            if len(arg) == 3:
                roles_exist_in_semlink = True
                (token_from, token_to), _, fn_role = arg
                positions = [token.data for token in tokens_to_positions[token_from:token_to+1]]
                if len(positions) == 0:
                    continue
                token_start, token_end = min(positions), max(positions) + 1
                matching_nodes = eds_primary_edges[token_start: token_end]
                if len(matching_nodes) == 0:
                    continue
                neighbour_node = min(matching_nodes, key=lambda i: i.length()).data
                matched_eds_annotated_edges[node.id, neighbour_node.id] = fn_role

        if roles_exist_in_semlink and len(matched_eds_annotated_edges) == 0:
            continue

        annotated_eds.append((eds_data, matched_frames, matched_eds_annotated_edges))

    return annotated_eds


def convert_eds_to_hetero_graph(eds_data, predicate2ix, attr2ix,
                                create_target=False,
                                frame_map=None, role_map=None,
                                frame2ix=None, role2ix=None):
    """
    Take an EDS object and construct a PyTorch graph.

    Parameters:
    eds_data (EDS): EDS object to create a graph for
    predicate2ix (dict): Lookup for predicate names
    attr2ix (dict): Lookup for attribute names
    create_target (bool): Create a target graph as well as input graph
    frame_map (dict): Partial map from node names to frames
    role_map (dict): Partial map from edges to roles
    frame2ix (dict): Map from frame names to indices
    role2ix (dict): Map from role names to indices

    Returns:
    graph (HeteroData): PyTorch EDS graph representation
    target_frames (Tensor): Frame target for each graph node
    target_roles (Tensor): Role target for each edge
    ix2node (dict): Mapping from graph node indices to node names
    """
    if create_target:
        assert frame_map is not None
        assert role_map is not None
        assert frame2ix is not None
        assert role2ix is not None

        frames = []
        roles = []

    nodes = []
    node2ix = {}
    ix2node = {}

    for node in eds_data.nodes:
        ix = len(node2ix)
        node2ix[node.id] = ix
        ix2node[ix] = node.id

        nodes.append(predicate2ix[node.predicate])
        if create_target:
            if node.id in frame_map:
                frames.append(frame2ix[frame_map[node.id]])
            else:
                frames.append(0)

    node_node_sources, node_node_targets = [], []
    node_edge_sources, node_edge_targets = [], []
    edge_node_sources, edge_node_targets = [], []
    edges = []

    for node in eds_data.nodes:
        for attr, neighbour in node.edges.items():
            source = node2ix[node.id]
            target = node2ix[neighbour]
            node_node_sources.append(source)
            node_node_targets.append(target)
            # node_node_targets.append(source)
            # node_node_sources.append(target)

            edge_id = len(edges)
            edges.append(attr2ix[attr])

            # edge_node_sources.append(edge_id)
            # edge_node_targets.append(source)
            # edge_node_sources.append(edge_id)
            # edge_node_targets.append(target)
            node_edge_sources.append(source)
            node_edge_targets.append(edge_id)
            edge_node_sources.append(edge_id)
            edge_node_targets.append(target)

            if create_target:
                if (node.id, neighbour) in role_map:
                    roles.append(role2ix[role_map[node.id, neighbour]])
                else:
                    roles.append(0)

    node_node_tensor = torch.tensor([node_node_sources, node_node_targets], dtype=torch.long)
    node_edge_tensor = torch.tensor([node_edge_sources, node_edge_targets], dtype=torch.long)
    edge_node_tensor = torch.tensor([edge_node_sources, edge_node_targets], dtype=torch.long)
    graph = HeteroData()
    graph["node"].x = torch.tensor(nodes, dtype=torch.long)
    graph["edge"].x = torch.tensor(edges, dtype=torch.long)
    graph["node", "true-edge", "node"].edge_index = node_node_tensor
    graph["edge", "edge-out", "node"].edge_index = edge_node_tensor
    graph["node", "edge-in", "edge"].edge_index = node_edge_tensor

    if create_target:
        target_frames = torch.tensor(frames, dtype=torch.long)
        target_roles = torch.tensor(roles, dtype=torch.long)

        return graph, (target_frames, target_roles), ix2node
    return graph, ix2node


class SemLinkDataset(Dataset):
    """
    SemLink Dataset.

    Contains EDS graphs annotated with the frame and roles of their primary
    predicates.
    """

    def __init__(self,
                 eds_sources,
                 ptb_sources,
                 semlink_path,
                 parse_load_path=None,
                 parse_save_path=None,
                 train_split=None,
                 dev_split=None,
                 test_split=None,
                 shuffle=True,
                 pre_compute_torch=True):
        """
        Construct a SemLink dataset.

        Parameters:
        eds_sources (str): Path to EDS source database
        semlink_path (str): Path to SemLink data file
        train_split (float): Percentage of data for training
        dev_split (float): Percentage of data for development
        test_split (float): Percentage of data for testing
        shuffle (bool): Shuffle data prior to taking splits
        """
        self.eds_sources = eds_sources
        self.semlink_path = semlink_path

        eds_dataset = load_eds_data_files(eds_sources)
        semlinks, semlink_lookup = load_semlink(semlink_path)

        indices = create_frame_and_role_indices(semlinks)
        self.frame2ix, self.ix2frame, self.role2ix, self.ix2role = indices

        if parse_load_path is None:
            sentence_to_ptb = parse_penn_treebank(ptb_sources, parse_save_path)
        else:
            sentence_to_ptb = load_penn_treebank(parse_load_path)

        self.eds_dataset = connect_eds_to_semlink(eds_dataset, sentence_to_ptb, semlink_lookup)

        print(f"Loaded {len(self.eds_dataset)} datapoints.")

        if shuffle:
            random.shuffle(self.eds_dataset)

        eds_graphs = [entry[0] for entry in self.eds_dataset]
        indices = create_predicate_and_attribute_indices(eds_graphs)
        self.predicate2ix, self.attr2ix = indices

        self.__split = None
        if train_split is not None:
            self.train_size = int(len(self.eds_dataset) * train_split)
            self.dev_size = int(len(self.eds_dataset) * dev_split)
            remainder = len(self.eds_dataset) - self.train_size - self.dev_size
            self.test_size = remainder

            self.train_offset = 0
            self.dev_offset = self.train_size
            self.test_offset = self.train_size + self.dev_size

        self.__view_indexed = False

        self.__pre_computed = None
        if pre_compute_torch:
            self.__pre_computed = self.__pre_compute()

    def view_data_indexed(self):
        """Switch data view mode to return indices for frames and roles."""
        self.__view_indexed = True

    def view_data_plain(self):
        """Switch data view mode to return strings for frames and roles."""
        self.__view_indexed = False

    def activate_train_set(self):
        """Load the train split."""
        self.__split = "train"

    def activate_dev_set(self):
        """Load the development split."""
        self.__split = "dev"

    def activate_test_set(self):
        """Load the test split."""
        self.__split = "test"

    def __pre_compute_single(self, eds_data, root_node, frame, roles):
        packed = convert_eds_to_hetero_graph(eds_data,
                                             self.predicate2ix,
                                             self.attr2ix,
                                             True,
                                             {root_node: frame},
                                             roles,
                                             self.frame2ix,
                                             self.role2ix)
        feature, (target_frame, target_roles), _ = packed
        root_node = target_frame.argmax()
        frame = torch.tensor(self.frame2ix[frame])

        return feature, root_node, frame, target_roles

    def __pre_compute(self):
        pre_computed = []
        for eds_data, (root_node, frame), roles in self.eds_dataset:
            pre_computed.append(self.__pre_compute_single(eds_data,
                                                          root_node,
                                                          frame,
                                                          roles))

        return pre_computed

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
        int: Length of the dataset
        """
        if self.__split is None:
            return len(self.eds_dataset)
        elif self.__split == "train":
            return self.train_size
        elif self.__split == "dev":
            return self.dev_size
        elif self.__split == "test":
            return self.test_size
        else:
            assert False, "invalid split selected"

    def __getitem__(self, index):
        """
        Return the item at the given index.

        Parameters:
        index (int): Index of the item to retrieve

        Returns:
        Features and targets for this dataset item.
        """
        if self.__split is None:
            offset = 0
        elif self.__split == "train":
            offset = self.train_offset
        elif self.__split == "dev":
            offset = self.dev_offset
        elif self.__split == "test":
            offset = self.test_offset
        else:
            assert False, "invalid split selected"

        index += offset

        if self.__pre_computed is not None and self.__view_indexed:
            return self.__pre_computed[index]

        if self.__view_indexed:
            eds_data, (root_node, frame), roles = self.eds_dataset[index]
            return self.__pre_compute_single(eds_data,
                                             root_node,
                                             frame,
                                             roles)

        eds_graph, frame, roles = self.eds_dataset[index]

        x = eds_graph
        y = frame, roles
        return x, y

    def compute_role_loss_weights(self):
        """
        Compute a weighting vector for role loss.

        This helps with the unbalanced data in roles, by allowing the loss
        function to penalise over-predicting the most common class.

        Returns:
        Tensor: vector of weights for each role
        """
        # We need all computed data here, so we pre-compute even if it wasn't
        # expressed upon creating the dataset
        if self.__pre_computed is None:
            self.__pre_computed = self.__pre_compute()

        role_loss_weights = torch.ones(len(self.role2ix))

        for feature, root, frame, roles in self.__pre_computed:
            adjacent_edges = torch.argwhere(feature["edge-in"].edge_index[0] == root)
            for role in roles[adjacent_edges]:
                role_loss_weights[role] += 1

        return 1.0 / role_loss_weights


def construct_semlink_dataset_from_config():
    """
    Load and create a SemLink dataset object from a configuration.

    Parameters:
    cfg: Configuration object

    Returns:
    SemLinkDataset: The loaded dataset
    """
    return SemLinkDataset(cfg.EDS_SOURCES_DIR,
                          cfg.PTB_SOURCES_DIR,
                          cfg.SEMLINK_DATA_FILE,
                          cfg.PTB_PARSE_LOAD_FILE,
                          cfg.PTB_PARSE_SAVE_FILE,
                          cfg.TRAIN_SPLIT,
                          cfg.DEV_SPLIT,
                          cfg.TEST_SPLIT,
                          cfg.SHUFFLE_BEFORE_SPLITTING)
