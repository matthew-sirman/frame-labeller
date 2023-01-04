"""
Data loader module.

Handles loading of EDS and SemLink databases
"""
import os
import delphin.codecs.eds as eds
import torch
import random
from torch.utils.data import Dataset

import config as cfg


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


def extract_eds_from_document(doc):
    """
    Given a document, find and extract the EDS section.

    Parameters:
    doc (str): The raw document text

    Returns:
    str: The EDS section of the document
    """
    eds_region = False
    eds = []

    lines = doc.splitlines()

    for line in lines:
        if line == "":
            continue
        if line[0] == "{":
            eds_region = True
        if eds_region:
            eds.append(line)
        if line[0] == "}":
            break

    return "\n".join(eds)


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
    if "=" not in arg_string:
        return loc, arg_string

    arg, arg_string = arg_string.split("=", 1)

    if ";" not in arg_string:
        return loc, arg

    _, fn_role = arg_string.split(";", 1)
    return loc, arg, fn_role


def load_semlink(semlink_path):
    """
    Load the SemLink data file.

    Parameters:
    semlink_path (str): File path to SemLink data file

    Returns:
    list: Parsed SemLink entries
    """
    semlinks = []
    with open(semlink_path) as semlink:
        for entry in semlink:
            entry = entry.strip().split(maxsplit=10)
            doc = entry[0]
            sentence = int(entry[1])
            token = int(entry[2])
            # Entries 3, 4 and 5 are ignored
            frame = entry[6]
            # Entries 7, 8 and 9 are ignored
            args = entry[10]

            args = [parse_arg(arg_string) for arg_string in args.split()]

            semlinks.append((doc, sentence, token, frame, args))

    return semlinks


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
    frame2ix = {}
    ix2frame = {}
    role2ix = {}
    ix2role = {}

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


def connect_eds_to_semlink(eds_dataset, semlink_dataset):
    """
    Pair entries in the SemLink database to EDS graphs.

    Parameters:
    eds_dataset (list): Entries from the EDS dataset
    semlink_dataset (list): Entries from the SemLink dataset

    Returns:
    list: Tuples annotating EDS graphs with their frames and roles
    """
    semlink_pairs = []

    for parse_file, sentence_id, _, frame, args in semlink_dataset:
        if frame in {"NF", "IN"}:
            continue

        major_id = os.path.splitext(parse_file)[0][-4:]
        deepbank_link = f"2{major_id}{sentence_id+1:03}"
        if deepbank_link not in eds_dataset:
            continue
        eds_raw_data = extract_eds_from_document(eds_dataset[deepbank_link])

        try:
            eds_data = eds.decode(eds_raw_data)
        except eds.EDSSyntaxError:
            continue

        eds_primary_edges = []
        for edge in eds_data[eds_data.top].edges:
            if edge in {"L-INDEX", "R-INDEX", "L-HNDL", "R-HNDL", "ARG"}:
                continue

            eds_primary_edges.append(edge)

        semlink_edges = []
        for arg in args:
            if len(arg) == 3 and arg[1][:3] == "ARG":
                semlink_edges.append((arg[1], arg[2]))

        if len(eds_primary_edges) != len(semlink_edges):
            continue

        eds_primary_edges.sort()
        semlink_edges.sort(key=lambda entry: entry[0])

        arg_data = {}

        for eds_arg, (_, role) in zip(eds_primary_edges, semlink_edges):
            arg_data[eds_arg] = role

        semlink_pairs.append((eds_data, frame, arg_data))

    return semlink_pairs


class SemLinkDataset(Dataset):
    """
    SemLink Dataset.

    Contains EDS graphs annotated with the frame and roles of their primary
    predicates.
    """

    def __init__(self,
                 eds_sources,
                 semlink_path,
                 train_split=None,
                 dev_split=None,
                 test_split=None,
                 shuffle=True):
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

        eds_dataset = dict(load_eds_data_files(eds_sources))
        semlinks = load_semlink(semlink_path)

        indices = create_frame_and_role_indices(semlinks)
        self.frame2ix, self.ix2frame, self.role2ix, self.ix2role = indices

        self.eds_dataset = connect_eds_to_semlink(eds_dataset, semlinks)

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

            dev_split_point = self.train_size + self.dev_size

            self.train_split = self.eds_dataset[:self.train_size]
            self.dev_split = self.eds_dataset[self.train_size:dev_split_point]
            self.test_split = self.eds_dataset[dev_split_point:]

        self.__view_indexed = False

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
        x (EDS): EDS graph for this entry
        y (str, dict) or (int, dict): Frame and roles for this entry
        """
        if self.__split is None:
            item = self.eds_dataset[index]
        elif self.__split == "train":
            item = self.train_split[index]
        elif self.__split == "dev":
            item = self.dev_split[index]
        elif self.__split == "test":
            item = self.test_split[index]
        else:
            assert False, "invalid split selected"

        eds_graph, frame, roles = item
        if self.__view_indexed:
            frame = torch.tensor(self.frame2ix[frame])
            roles = {arg: torch.tensor(self.role2ix[roles[arg]]) for arg in roles}

        x = eds_graph
        y = frame, roles
        return x, y


def construct_semlink_dataset_from_config():
    """
    Load and create a SemLink dataset object from a configuration.

    Parameters:
    cfg: Configuration object

    Returns:
    SemLinkDataset: The loaded dataset
    """
    return SemLinkDataset(cfg.EDS_SOURCES_DIR,
                          cfg.SEMLINK_DATA_FILE,
                          cfg.TRAIN_SPLIT,
                          cfg.DEV_SPLIT,
                          cfg.TEST_SPLIT,
                          cfg.SHUFFLE_BEFORE_SPLITTING)
