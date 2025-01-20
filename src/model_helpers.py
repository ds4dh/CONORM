import os

# Save the original value of TF_CPP_MIN_LOG_LEVEL
original_tf_cpp_min_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL')

# Set TF_CPP_MIN_LOG_LEVEL to 3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Python Standard Library
from collections import Counter, defaultdict, OrderedDict
from datetime import datetime, timedelta
import json
import logging
import pickle
import time
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import re

# Third-party Libraries
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

# Transformers
from transformers import (
    AutoModel, 
    AutoModelForTokenClassification, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup
)
from torch.optim import Adam, AdamW

# Set TF_CPP_MIN_LOG_LEVEL back to its original value
if original_tf_cpp_min_log_level is not None:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_tf_cpp_min_log_level
else:
    del os.environ['TF_CPP_MIN_LOG_LEVEL']


###############################
####### DATA MANAGEMENT #######
###############################

def save_list(my_list: List, filename: str) -> None:
    """
    Save a list to a pickle file.

    Args:
        my_list (List): The list to be saved.
        filename (str): The name of the file to save the list to.

    Raises:
        IOError: If an error occurs while writing to the file.

    Returns:
        None
    """
    try:
        with open(filename, 'wb') as file:
            pickle.dump(my_list, file)
    except IOError as e:
        print(f"An error occurred while saving the list: {e}")


def load_data(data_dir: str,
              data_groups: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    """
    Load the data from the specified files.

    Args:
        data_dir: Directory where the data files are stored.
        data_groups: List of data groups (train, val, test).

    Returns:
        Tuple of dictionaries with sentences and labels.
    """
    data_path = {data_group: os.path.join(
        data_dir, f"{data_group}.txt") for data_group in data_groups}
    sentences = {}
    starts = {}
    ends = {}
    file_names = {}
    labels = {}
    for data_group in data_groups:
        sentences[data_group] = []
        starts[data_group] = []
        ends[data_group] = []
        file_names[data_group] = []
        labels[data_group] = []
        with open(data_path[data_group], 'r', encoding='utf-8') as f:
            file = f.read()
        i = 0
        for sentence in file.strip().split('\n\n'):
            s = []
            starts[data_group].append([])
            ends[data_group].append([])
            file_names[data_group].append([])
            labels[data_group].append([])
            for x in sentence.split('\n'):
                t_splits = x.split('\t')
                word = t_splits[0]
                start, end, file_name = t_splits[-3:]
                tag = t_splits[1:-3]
                s.append(word)
                starts[data_group][-1].append(int(start))
                ends[data_group][-1].append(int(end))
                file_names[data_group][-1].append(file_name)
                labels[data_group][-1].append(tag)
            sentences[data_group].append(" ".join([w for w in s]))
            i += 1
        logging.warning(" " + data_group + ' number of instances = ' + str(i))

    tags_vals = list(
        chain(*(list(chain(*[labels[data_group] for data_group in data_groups])))))
    transposed_labels = zip(*tags_vals)
    unique_label_per_column = []
    for column in transposed_labels:
        column = [i.split("-")[-1] for i in column]
        unique_label = [x for x in list(set(column)) if x != 'O'][0]
        unique_label_per_column.append(unique_label)

    return sentences, labels, starts, ends, file_names, unique_label_per_column


def create_tag2idx(columns: List[str]) -> Dict[str, int]:
    """Create a dictionary mapping tag pairs to indices.

    Args:
        columns: A list of column names.

    Returns:
        A dictionary mapping tag pairs to indices.
    """

    # Create a set of unique tags
    unique_set = list(set(columns))

    # Create a dictionary mapping tag pairs to indices
    tag2idx = {}
    for tag1 in unique_set:
        idx = 0
        for tag2 in ["B", "I", "X", "O"]:
            tag2idx[f"{tag2}-{tag1}"] = idx
            idx += 1

    tag2idx["[PAD]"] = 3

    return tag2idx


def create_path(folder_path: str,
                pretrained_model: Optional[str] = None) -> str:
    """
    Create a path for saving logs based on the inputs.

    Args:
        folder_path (str): The folder path to use for saving the logs.
        pretrained_model (str, optional): The pretrained model to use. Defaults to None.

    Returns:
        str: The path for saving the logs.
    """
    # Get the current date and time
    date = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Get the language data and entities
    lang_data = "_".join(folder_path.split("/")[2:4])

    # Define model name
    model_name = pretrained_model + "_" + date
    model_name = model_name.replace("/", "_")

    # Define the path to save the logs
    PATH = os.path.join('.', 'logs', lang_data, model_name)

    return PATH


def save_config(config: Dict,
                filename: str) -> None:
    """
    Save a configuration dictionary to a JSON file.

    Args:
        config (Dict): The configuration dictionary to save.
        filename (str): The name of the file to save the configuration to.

    Returns:
        None
    """
    # Open the specified file in write mode
    with open(filename, 'w') as f:
        # Use json.dump to write the config dictionary to the file
        json.dump(config, f, indent=4)


def align_data(data_groups: List[str],
               sentences: Dict[str, List[str]],
               labels: Dict[str, List[str]],
               starts: Dict[str, List[int]],
               ends: Dict[str, List[int]],
               file_names: Dict[str, List[str]],
               tokenizer,
               MAX_LEN: int) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray], Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[str]]]:
    """
    Align the data to be compatible with transformer-encoders from HuggingFace.

    Args:
        data_groups: A list of strings representing the groups of data.
        sentences: A dictionary where the keys are the data groups and the values are lists of sentences.
        labels: A dictionary where the keys are the data groups and the values are lists of labels.
        starts: A dictionary where the keys are the data groups and the values are lists of starting character indices for each token in the sentences.
        ends: A dictionary where the keys are the data groups and the values are lists of ending character indices for each token in the sentences.
        file_names: A dictionary where the keys are the data groups and the values are lists of file names.
        tokenizer: A Tokenizer object to tokenize the words.
        MAX_LEN: An integer representing the maximum length of the sequences.

    Returns:
        A tuple of five dictionaries where the keys are the data groups and the values are lists of sentences, labels, aligned text and aligned labels.
    """
    aligned_labels = {}
    aligned_text = {}
    aligned_subword_dummies = {}
    aligned_starts = {}
    aligned_ends = {}
    aligned_file_names = {}

    # Iterate through each group of data
    for data_group in data_groups:

        # Initialize empty lists to store the new data for this group
        aligned_labels[data_group] = []
        aligned_text[data_group] = []
        aligned_subword_dummies[data_group] = []
        aligned_starts[data_group] = []
        aligned_ends[data_group] = []
        aligned_file_names[data_group] = []

        # Initialize counters for number of too long sequences and misalignments
        too_big_count = 0
        missalign_count = 0

        # Iterate through each sentence and its corresponding labels
        for sent, tags, start, end, file_name in zip(tqdm(sentences[data_group], leave=False, desc=f"{data_group} text/start/end/file_name alignment"), labels[data_group], starts[data_group], ends[data_group], file_names[data_group]):

            BERT_subwords_dummy = []
            BERT_texts = []
            BERT_labels = np.array([])
            BERT_starts = []
            BERT_ends = []
            BERT_file_names = []

            # Check that the number of words in the sentence matches the number of labels
            if len(sent.split()) != len(tags):
                missalign_count += 1
                continue

            for word, tag, st, en, fn in zip(sent.split(), tags, start, end, file_name):

                sub_words = tokenizer.tokenize(word)
                sub_words_fn = [fn for _ in sub_words]
                sub_words_st = [st] + [-1] * (len(sub_words) - 1)
                sub_words_en = [en] + [-1] * (len(sub_words) - 1)
                tags_ = np.array(
                    [[t if i == 0 else f"X-{t.split('-')[-1]}" for i, _ in enumerate(sub_words)] for t in tag])

                BERT_subwords_dummy.extend([0] + [1] * (len(sub_words) - 1))
                BERT_texts.extend(sub_words)
                BERT_labels = tags_ if BERT_labels.size == 0 else np.hstack(
                    [BERT_labels, tags_])
                BERT_starts.extend(sub_words_st)
                BERT_ends.extend(sub_words_en)
                BERT_file_names.extend(sub_words_fn)

            if len(BERT_texts) > MAX_LEN:
                BERT_texts = BERT_texts[:MAX_LEN]
                BERT_labels = BERT_labels[:, :MAX_LEN]
                BERT_subwords_dummy = BERT_subwords_dummy[:MAX_LEN]
                BERT_starts = BERT_starts[:MAX_LEN]
                BERT_ends = BERT_ends[:MAX_LEN]
                BERT_file_names = BERT_file_names[:MAX_LEN]
                too_big_count += 1

            aligned_subword_dummies[data_group].append(BERT_subwords_dummy)
            aligned_text[data_group].append(BERT_texts)
            aligned_labels[data_group].append(BERT_labels)
            aligned_starts[data_group].append(BERT_starts)
            aligned_ends[data_group].append(BERT_ends)
            aligned_file_names[data_group].append(BERT_file_names)

        # Log the number of too long sequences and misalignments
        logging.warning(
            f' Too long instances in {data_group} = {too_big_count} (truncated to {MAX_LEN} tokens)')
        logging.warning(
            f' Misaligned text/label instances in {data_group} = {missalign_count}')

    return aligned_text, aligned_labels, aligned_subword_dummies, aligned_starts, aligned_ends, aligned_file_names


class Numericaliser:
    def __init__(self, vocab: List[str]) -> None:
        """
        Initialize the Numericaliser class with a vocabulary list

        Args:
        - vocab (List[str]): list of vocabulary

        """
        self.vocab = vocab + ["[UNK]"]

    def s2n(self, string: str) -> int:
        """
        Convert a string to its corresponding numerical value based on the vocabulary

        Args:
        - string (str): string to be converted

        Returns:
        - int: numerical value of the string

        """
        return self.vocab.index(string)

    def n2s(self, numerical: int) -> str:
        """
        Convert a numerical value to its corresponding string based on the vocabulary

        Args:
        - numerical (int): numerical value to be converted

        Returns:
        - str: string value of the numerical

        """
        return self.vocab[numerical]


def pad_sequences(sequences: List[Union[List, Tuple]],
                  maxlen: int = None,
                  dtype: str = "int32",
                  padding: str = "pre",
                  truncating: str = "pre",
                  value: Union[int, float, str] = 0.0) -> np.ndarray:
    """
    Pads sequences to a common length.

    Args:
        sequences: A list of sequences, where each sequence is a list or tuple of
            values.
        maxlen: The maximum length of the sequences. If not specified, the maximum
            length of the sequences in `sequences` is used.
        dtype: The data type of the returned array.
        padding: The position to pad the sequences ("pre" or "post").
        truncating: The position to truncate the sequences ("pre" or "post").
        value: The value used to pad the sequences.

    Returns:
        A numpy array of shape `(len(sequences), maxlen, sample_shape)`, where
        `sample_shape` is the shape of the first non-empty sequence in `sequences`.
    """

    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError as e:
            raise ValueError(
                "`sequences` must be a list of iterables. "
                f"Found non-iterable: {str(x)}"
            ) from e

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(
        dtype, np.unicode_
    )
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError(
            f"`dtype` {dtype} is not compatible with `value`'s type: "
            f"{type(value)}\nYou should set `dtype=object` for variable length "
            "strings."
        )

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at "
                f"position {idx} is different from expected shape "
                f"{sample_shape}"
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x


def create_dataloader(data_group: str,
                      aligned_text: Dict[str, List[List[str]]],
                      aligned_labels: Dict[str, List[List[str]]],
                      aligned_subword_dummy: Dict[str, List[List[int]]],
                      aligned_starts: Dict[str, List[List[int]]],
                      aligned_ends: Dict[str, List[List[int]]],
                      aligned_file_names: Dict[str, List[List[str]]],
                      tokenizer,
                      tag2idx: Dict[str, int],
                      MAX_LEN: int,
                      BATCH_SIZE: int,) -> Tuple[DataLoader, Numericaliser, Numericaliser]:
    """
    Create data loader for the given data_group.

    Args:
    - data_group: a string representing the group of data.
    - aligned_text: Dictionary where the keys are the data groups and the values are lists of aligned text.
    - aligned_labels: Dictionary where the keys are the data groups and the values are lists of aligned labels.
    - aligned_subword_dummy: Dictionary where the keys are the data groups and the values are lists of subword dummies.
    - aligned_starts: Dictionary where the keys are the data groups and the values are lists of aligned starts.
    - aligned_ends: Dictionary where the keys are the data groups and the values are lists of aligned ends.
    - aligned_file_names: Dictionary where the keys are the data groups and the values are lists of aligned file names.
    - tokenizer: Tokenizer object to tokenize the words.
    - tag2idx: Dictionary where the keys are the labels and the values are corresponding ids.
    - MAX_LEN: Maximum length of the sequences.
    - BATCH_SIZE: Batch size for the data loader.

    Returns:
    - Tuple of data loader, Numericaliser for file names, and Numericaliser for dummy words.
    """
    # Convert the aligned text into input ids and pad them to the maximum length
    input_ids = torch.tensor(
        pad_sequences(
            [tokenizer.convert_tokens_to_ids(txt) for txt in tqdm(
                aligned_text[data_group], leave=False, desc=f"Creating {data_group} input_ids")],
            maxlen=MAX_LEN,
            dtype="long",
            truncating="post",
            padding="post"
        )
    )

    # Convert the aligned labels into tag ids and pad them to the maximum length
    tags = torch.tensor(
        np.array(
            [pad_sequences([[tag2idx[i] for i in l] for l in lab],
                           maxlen=MAX_LEN,
                           value=tag2idx["[PAD]"],
                           padding="post",
                           dtype="long",
                           truncating="post"
                           )
                for lab in tqdm(aligned_labels[data_group], leave=False, desc=f"Creating {data_group} padded sequences")
             ]
        )
    ).transpose(-1, -2)

    # Create attention masks by checking which input ids are non-zero
    attention_masks = torch.tensor(
        [
            [float(i > 0) for i in ii]
            for ii in tqdm(
                input_ids,
                leave=False,
                desc=f"Creating {data_group} mask tensors"
            )
        ]
    )

    # Create subword dummies
    subword_dummies = torch.tensor(
        pad_sequences(
            aligned_subword_dummy[data_group],
            maxlen=MAX_LEN,
            value=0,
            dtype="long",
            truncating="post",
            padding="post"
        )
    )

    # Create starts
    starts = torch.tensor(
        pad_sequences(
            aligned_starts[data_group],
            maxlen=MAX_LEN,
            value=0,
            dtype="long",
            truncating="post",
            padding="post"
        )
    )

    # Create ends
    ends = torch.tensor(
        pad_sequences(
            aligned_ends[data_group],
            maxlen=MAX_LEN,
            value=0,
            dtype="long",
            truncating="post",
            padding="post"
        )
    )

    # Create file_names
    str2num_file_names = Numericaliser(
        list(set(list(chain(*aligned_file_names[data_group]))))
    )

    file_names = torch.tensor(
        pad_sequences(
            [
                [
                    str2num_file_names.s2n(i)
                    for i in j
                ]
                for j in tqdm(aligned_file_names[data_group], leave=False, desc=f"Creating {data_group} input_ids")
            ],
            maxlen=MAX_LEN,
            dtype="long",
            truncating="post",
            padding="post",
            value=str2num_file_names.s2n("[UNK]")
        )
    )

    # Create tensor dataset and dataloader for the given data_group
    data = TensorDataset(input_ids, attention_masks, tags,
                         subword_dummies, starts, ends, file_names)
    sampler = RandomSampler(
        data) if data_group == 'train' else SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)

    return dataloader, str2num_file_names


########################
####### MODELING #######
########################


class LinearClassifier(torch.nn.Module):
    """
    A simple linear classifier module.

    Args:
        in_features (int): the number of input features for the linear layer
        out_features (int): the number of output features for the linear layer

    """

    def __init__(self, in_features: int, out_features: int):

        super().__init__()
        self.output = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear classifier.

        Args:
            x (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: the output of the linear layer
        """

        x = self.output(x)

        return x


class MyAutoModelForTokenClassification(torch.nn.Module):

    class ClassificationLayer(torch.nn.Module):
        def __init__(self, embedding_dim: int, num_labels: int):
            """
            Initialize ClassificationLayer.

            Args:
                embedding_dim (int): The dimension of the embeddings.
                num_labels (int): The number of labels.
            """
            super().__init__()
            self.classifier = LinearClassifier(embedding_dim + 1, num_labels)
            self.ce = torch.nn.CrossEntropyLoss()

    def __init__(self,
                 pretrained_model_name_or_path: str,
                 tag2idx: dict,
                 columns: List[str]):
        """
        Initialize MyAutoModelForTokenClassification.

        Args:
            pretrained_model_name_or_path (str): Pretrained model name or path.
            tag2idx (dict): Dictionary mapping tags to indices.
            columns (List[str]): List of column names.
        """
        super().__init__()

        self.num_labels = len(set(tag2idx.values()))
        AutoModel = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path, num_labels=self.num_labels, ignore_mismatched_sizes=True)
        embedding_dim = AutoModel.classifier.in_features

        self.language_model = AutoModel.base_model
        self.dropout = AutoModel.dropout

        self.classification_layers = torch.nn.ModuleList([
            self.ClassificationLayer(embedding_dim, self.num_labels)
            for _ in range(len(columns))
        ])

    def forward(self,
                input_ids: torch.Tensor,
                subword_dummies: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for MyAutoModelForTokenClassification.

        Args:
            input_ids (torch.Tensor): Tensor of input IDs.
            subword_dummies (torch.Tensor): Tensor of subword dummies.
            attention_mask (torch.Tensor): Tensor of attention masks.
            labels (Optional[torch.Tensor]): Optional tensor of true labels.

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: Tuple containing loss and output.
        """

        seq = self.language_model(
            input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(seq.last_hidden_state)
        seq = [seq for layer in self.classification_layers]
        seq = [torch.cat((sub_seq, subword_dummies.unsqueeze(-1)), dim=-1)
               for sub_seq in seq]
        logits = [layer.classifier(sub_seq) for layer, sub_seq in zip(
            self.classification_layers, seq)]

        if self.training:
            flat_mask = attention_mask.view(-1)
            flat_logits = [
                logits[i].view(-1, self.num_labels)[flat_mask != 0, :] for i in range(len(logits))]
            flat_labels = [
                labels[:, :, i].view(-1)[flat_mask != 0] for i in range(len(logits))]
            ce_loss = torch.mean(torch.stack([
                layer.ce(flat_logits[i], flat_labels[i])
                for i, layer in enumerate(self.classification_layers)]))
            return (ce_loss, None)
        else:
            if labels is not None:
                flat_mask = attention_mask.view(-1)
                flat_logits = [
                    logits[i].view(-1, self.num_labels)[flat_mask != 0, :] for i in range(len(logits))]
                flat_labels = [
                    labels[:, :, i].view(-1)[flat_mask != 0] for i in range(len(logits))]
                ce_loss = torch.mean(torch.stack([
                    layer.ce(flat_logits[i], flat_labels[i])
                    for i, layer in enumerate(self.classification_layers)]))
                return (ce_loss, logits)
            else:
                return (None, logits)


def train_step(dataloader: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               device: torch.device) -> float:
    """
    Trains a model on given dataloader.

    Parameters:
        dataloader (torch.utils.data.DataLoader): DataLoader containing the training dataset.
        model (nn.Module): Pytorch model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler for the optimizer.
        device (torch.device): Device to move the data and model to.

    Returns:
        avg_loss (float): Average loss.
    """
    total_loss = 0
    total_steps = 0
    for step, batch in enumerate(tqdm(dataloader, leave=False, desc="Training")):

        # For training, take only first 4 elements from dataloader.
        batch = tuple(t.to(device) for t in batch[:4])
        input_ids, input_mask, labels, subword_dummies = batch

        loss = model(input_ids, subword_dummies,
                     attention_mask=input_mask, labels=labels)[0]
        loss.backward()
        total_loss += loss.item()
        total_steps += 1
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    avg_loss = total_loss/total_steps

    return avg_loss


def col2dict_multilabel(columns: List[str]) -> Tuple[List[Dict[str, int]], List[Dict[int, str]]]:
    """
    Converts a list of named entity column names to two dictionaries mapping named entity labels to integer indices 
    and vice versa. Each named entity label is represented by a prefix of either "B-" or "I-", followed by the entity 
    type. The "B-" prefix is used to mark the beginning of a named entity, while the "I-" prefix is used to mark the 
    continuation of a named entity.

    Args:
        columns: A list of column names representing named entity types.

    Returns:
        A tuple containing two lists of dictionaries:
            - A list of dictionaries mapping named entity labels to integer indices.
            - A list of dictionaries mapping integer indices to named entity labels.

    """
    forward_list = [{'B-'+item: 0, 'I-'+item: 1, 'X': 2, 'O': 3}
                    for item in columns]
    reverse_list = [{value: key for key, value in dict_item.items()}
                    for dict_item in forward_list]
    return forward_list, reverse_list


def not_mask_subwords_index(x: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
    """
    Given a subword mask and a sequence, return the indexes of the elements that are not subwords.

    Args:
    - x (np.ndarray): sequence with subword dummy variables.
    - mask (np.ndarray): binary mask indicating subwords positions.

    Returns:
    - np.ndarray: indexes of elements that are not subwords.
    """
    return np.where(x[mask == 1] != 1)[0]


def filtered_logit_sequence(subword_dummies: np.ndarray,
                            logits: np.ndarray,
                            input_mask: np.ndarray,
                            tag2idx: Dict[str, int]) -> List[List[int]]:
    """
    Given a subword mask and logits, remove the subword positions and return the filtered logit sequence.

    Args:
    - subword_dummies (np.ndarray): binary mask indicating subwords positions.
    - logits (np.ndarray): logits from the model.
    - input_mask (np.ndarray): binary mask indicating valid input positions.
    - tag2idx (Dict[str, int]): mapping from labels to integer indices.

    Returns:
    - List[List[int]]: filtered logit sequence.
    """
    output = []
    ind = tag2idx["X"]
    for sd, lo, ma in zip(subword_dummies, logits, input_mask):
        index = not_mask_subwords_index(sd, ma)
        lo[:, ind] = -float("inf")
        lo_filtered = lo[index, :].argmax(-1).tolist()
        output.append(lo_filtered)
    return output


def filtered_true_sequence(subword_dummies: np.ndarray,
                           true: np.ndarray,
                           input_mask: np.ndarray) -> List[List[int]]:
    """
    Given a subword mask and logits, remove the subword positions and return the filtered true sequence.

    Args:
    - subword_dummies (np.ndarray): binary mask indicating subwords positions.
    - true (np.ndarray): true labels of the input.
    - input_mask (np.ndarray): binary mask indicating valid input positions.

    Returns:
    - List[List[int]]: filtered true sequence.
    """
    output = []
    for sd, tr, ma in zip(subword_dummies, true, input_mask):
        index = not_mask_subwords_index(sd, ma)
        tr_filtered = tr[index].tolist()
        output.append(tr_filtered)
    return output


def infer(dataloader: torch.utils.data.DataLoader,
          model: torch.nn.Module,
          device: torch.device,
          columns: List[str],
          tag2idx: Dict[int, str],
          str2num_file_names: Numericaliser) -> List[List[Tuple[str, List[str], int, int, str]]]:
    """
    Performs inference on a given dataloader using a trained model.

    Args:
    - dataloader (torch.utils.data.DataLoader): DataLoader object for the data to be inferred
    - model (torch.nn.Module): PyTorch model to be used for inference
    - device (torch.device): the device to run the inference on (GPU or CPU)
    - columns (List[str]): a list of column names used in the dataset
    - tag2idx (Dict[int, str]): a dictionary that maps the tag index to its label
    - str2num_file_names (Numericaliser): a Numericaliser object to convert file names from string to numerical values

    Returns:
    - List[List[Tuple[str, List[str], int, int, str]]]: a list of tuples that contain the predicted values for each column

    """

    num_heads = len(model.classification_layers)
    columns_tag2idx, columns_idx2tag = col2dict_multilabel(columns)

    Predictions = [[] for _ in range(num_heads)]
    Starts = []
    Ends = []
    File_names = []

    for step, batch in enumerate(tqdm(dataloader, leave=False, desc="Inferring")):

        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, tags, subword_dummies, starts, ends, file_names = batch

        # Set model in evaluation mode and disable gradient calculation
        with torch.no_grad():
            evaluation_result = model(
                input_ids, subword_dummies, attention_mask=input_mask, labels=None)
            logits = evaluation_result[1]

        logits = [logit.detach().cpu().numpy() for logit in logits]
        subword_dummies = subword_dummies.cpu().numpy()
        input_mask = input_mask.cpu().numpy()
        starts = starts.cpu().numpy()
        ends = ends.cpu().numpy()
        file_names = file_names.cpu().numpy()

        starts = filtered_true_sequence(subword_dummies, starts, input_mask)
        ends = filtered_true_sequence(subword_dummies, ends, input_mask)
        file_names = filtered_true_sequence(
            subword_dummies, file_names, input_mask)

        for layer_idx in range(num_heads):

            preds = filtered_logit_sequence(
                subword_dummies, logits[layer_idx], input_mask, columns_tag2idx[layer_idx])
            preds = [[columns_idx2tag[layer_idx][i] for i in j] for j in preds]
            assert len(preds) == len(starts) == len(ends) == len(file_names)
            Predictions[layer_idx].append(preds)

        Starts.append(starts)
        Ends.append(ends)
        File_names.append(file_names)

    Predictions = [list(chain(*i)) for i in Predictions]
    Starts = list(chain(*Starts))
    Ends = list(chain(*Ends))
    File_names = list(chain(*File_names))

    output = []
    for P in Predictions:
        results = []
        for j, k, l, m in zip(P, Starts, Ends, File_names):
            sub_results = []
            for prediction, start, end, file_name in zip(j, k, l, m):
                sub_results.append(
                    ("", prediction, start, end, str2num_file_names.n2s(file_name)))
            results.append(sub_results)
        output.append(results)

    return output


def parse_summary(file_path: str) -> Dict[str, Dict[str, float]]:
    """
    Parses the summary output file generated by the BRATEval-0.0.2-SNAPSHOT.jar command line tool and returns
    a dictionary containing evaluation metrics for each entity type.

    Args:
        file_path: The path to the summary output file.

    Returns:
        A dictionary where keys are entity types and values are dictionaries containing the following metrics:
        TP: True Positives
        FP: False Positives
        FN: False Negatives
        Precision: Precision score
        Recall: Recall score
        F1: F1 score

    Raises:
        FileNotFoundError: If the summary output file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Summary file not found at {file_path}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the start of the summary section
    summary_start_index = -1
    for i, line in enumerate(lines):
        if line.startswith('Summary'):
            summary_start_index = i + 2  # skip header row
            break

    # If summary section not found, return empty dictionary
    if summary_start_index == -1:
        return {}

    # Extract summary section lines and remove empty lines
    summary_lines = [line.strip() for line in lines[summary_start_index:]]
    summary_lines = [line for line in summary_lines if line]

    # Parse summary section and store metrics in a dictionary
    summary = {}
    for line in summary_lines:
        fields = line.split('\t')
        summary[fields[0]] = {
            'TP': int(fields[1]),
            'FP': int(fields[2]),
            'FN': int(fields[3]),
            'Precision': float(fields[4]),
            'Recall': float(fields[5]),
            'F1': float(fields[6])
        }

    return summary


def evaluate(pred_folder: str,
             reference_data_folder: str) -> Dict[str, Dict[str, float]]:
    """
    Executes the BRATEval-0.0.2-SNAPSHOT.jar command line tool with the provided paths and returns
    the parsed summary output.

    Args:
        pred_folder: The path to the prediction folder.
        reference_data_folder: The path to the reference data folder.

    Returns:
        A dictionary where keys are entity types and values are dictionaries containing the following metrics:
        TP: True Positives
        FP: False Positives
        FN: False Negatives
        Precision: Precision score
        Recall: Recall score
        F1: F1 score
    """
    # Execute BRATEval-0.0.2-SNAPSHOT.jar command
    bashCommand = f"""java -cp BRATEval-0.0.2-SNAPSHOT.jar au.com.nicta.csp.brateval.CompareEntities {pred_folder} {reference_data_folder} true > {"/".join(pred_folder.split("/")[:-1])}/Summary"""
    os.system(bashCommand)

    # Parse summary output
    summary = parse_summary(
        f"""{"/".join(pred_folder.split("/")[:-1])}/Summary""")

    return summary


def sent2doc(list1: List[List[Any]]) -> List[List[Any]]:
    """
    Groups a list of sublists into a list of sublists based on the value of the fifth element.

    Args:
        list1: A list of sublists, where each sublist contains at least five elements.

    Returns:
        A list of sublists, where each sublist contains all the elements from `list1`
        with the same value of the fifth element.
    """
    # Create a dictionary to store the sublists based on the fifth element
    dict_ = defaultdict(list)
    for sublist in list1:
        for item in sublist:
            dict_[item[4]].append(item)
    # Convert the dictionary values to sublists and return the final list
    return [value for value in dict_.values()]


def b_decision(x: List[str]) -> str:
    """
    Returns the tag suffix after the hyphen for the first non-"O" tag in a list.

    Args:
        x: A list of tags.

    Returns:
        The tag suffix after the hyphen for the first non-"O" tag in `x`.
    """
    return x[0].split("-")[1]


def post_process_heuristic(pred: List[Tuple[str, str, int, int, str]]) -> Tuple[List[Tuple[str, str, int, int]], str]:
    """
    Groups predictions by tag and applies the b_decision strategy to each group.

    Args:
        pred: A list of predictions, where each prediction is a tuple containing
              a word, a tag, a start position, an end position, and a file name.

    Returns:
        A tuple containing a list of new predictions and the file name of the last
        prediction in `pred`.
    """
    groups = np.zeros(len(pred))
    prev_tag = "O"
    group_num = 0
    for counter, (text, tag, start, end, file_name) in enumerate(pred):
        if tag.split("-")[0] != "O":
            if prev_tag.split("-")[-1] != tag.split("-")[-1]:
                group_num += 1
            groups[counter] = group_num
        prev_tag = "O" if tag.split("-")[0] == "O" else tag

    groups = [np.where(groups == i)[0].tolist() for i in set(groups) if i != 0]
    new_pred = []
    for group in groups:
        start = pred[group[0]][2]
        end = pred[group[-1]][3]
        file_name = pred[group[-1]][4]
        tags = [pred[ind][1] for ind in group]
        text = " ".join(pred[ind][0] for ind in group)
        new_pred.append((text.strip(), b_decision(tags), start, end))
    return new_pred, file_name


def most_frequent(tags: List[str]) -> str:
    """
    Returns the most frequent tag from a list of tags. In case of a tie, one of the most frequent tags is chosen arbitrarily.

    Args:
        tags: A list of tags.

    Returns:
        The most frequent tag in `tags`.
    """
    # Count the frequency of each tag
    tag_counts = Counter(tags)

    # Find the most common tag(s)
    most_common_tags = tag_counts.most_common()

    if not most_common_tags:
        return "O"  # Default to "O" if no tags are present

    # Return the tag of the first most common element
    return most_common_tags[0][0].split("-")[1]


def post_process_classic(pred: List[Tuple[str, str, int, int, str]]) -> Tuple[List[Tuple[str, str, int, int]], str]:
    """
    Groups predictions by tag and applies the most_frequent strategy to each group, with improved handling of discontinuous entities.

    Args:
        pred: A list of predictions, where each prediction is a tuple containing
              a word, a tag, a start position, an end position, and a file name.

    Returns:
        A tuple containing a list of new predictions and the file name of the last prediction in `pred`.
    """

    if not pred:
        return [], ""

    new_pred = []
    current_group = []
    last_tag = None
    file_name = ""

    for word, tag, start, end, file_name in pred:
        tag_type = tag.split("-")[-1] if tag != "O" else "O"

        if tag.startswith("B-") or (tag == "O" and current_group):
            if current_group:
                new_pred.append(process_group(current_group, most_frequent))
                current_group = []
            if tag.startswith("B-"):
                current_group = [(word, tag, start, end)]

        elif tag.startswith("I-") and tag_type == last_tag:
            current_group.append((word, tag, start, end))

        last_tag = tag_type

    if current_group:
        new_pred.append(process_group(current_group, most_frequent))

    return new_pred, file_name


def process_group(group, decision_strat):
    """
    Processes a group of predictions to create a single, corrected prediction.

    Args:
        group: A list of tuples, each containing a word, tag, start position, end position.
        decision_strat: A function that takes a list of tags and returns a new tag.

    Returns:
        A tuple containing the concatenated text, corrected tag, start position, and end position.
    """
    text = " ".join([word for word, _, _, _ in group])
    tags = [tag for _, tag, _, _ in group]
    start = group[0][2]
    end = group[-1][3]
    corrected_tag = decision_strat(tags)
    return (text, corrected_tag, start, end)


def write_ann(ann: List[Tuple[str, str, int, int]],
              ID: str,
              text_folder_path: str,
              folder: str = "",
              counter: int = 0,
              mode: str = 'w',
              split_brat_on_newlines: bool = False) -> int:
    """
    Writes annotations to a BRAT .ann file, adjusting for whitespace and handling newlines.

    Args:
        ann: A list of annotations.
        ID: File ID.
        text_folder_path: Path to original text files.
        folder: Path for saving .ann files.
        counter: Starting counter for annotations.
        mode: File mode ('w' for write, 'a' for append).
        split_brat_on_newlines: Flag to split annotations on newline characters.

    Returns:
        New counter value after writing annotations.
    """
    with open(os.path.join(text_folder_path, f"{ID}.txt"), 'r', encoding='utf-8') as f:
        original_text = f.read()

    to_write = []
    for i, (text, tag, start, end) in enumerate(ann):
        entity_text = original_text[start:end]
        segments = []

        if split_brat_on_newlines:
            prev_pos = 0
            for pos, char in enumerate(entity_text):
                if char == "\n":
                    segment_text = entity_text[prev_pos:pos].strip()
                    if segment_text:
                        new_start = start + prev_pos + entity_text[prev_pos:pos].find(segment_text)
                        new_end = new_start + len(segment_text)
                        segments.append((new_start, new_end))
                    prev_pos = pos + 1

            # Add the last segment if there's text after the last newline
            if prev_pos < len(entity_text):
                segment_text = entity_text[prev_pos:].strip()
                if segment_text:
                    new_start = start + prev_pos + entity_text[prev_pos:].find(segment_text)
                    new_end = new_start + len(segment_text)
                    segments.append((new_start, new_end))

            # Combine segments, treating consecutive newlines as single breaks
            combined_segments = []
            for s, e in segments:
                if not combined_segments or s > combined_segments[-1][1]:
                    combined_segments.append((s, e))
                else:
                    combined_segments[-1] = (combined_segments[-1][0], e)

            new_start_end = ';'.join([f"{s} {e}" for s, e in combined_segments])
            trimmed_text = " ".join(original_text[s:e] for s, e in combined_segments)
            to_write.append(f"T{counter+i+1}" + "\t" + str(tag) + " " + new_start_end + "\t" + trimmed_text.replace("\n", " ") + "\n")
        else:
            trimmed_text = entity_text.strip()
            new_start = start + entity_text.find(trimmed_text)
            new_end = new_start + len(trimmed_text)
            to_write.append(f"T{counter+i+1}" + "\t" + str(tag) + f" {new_start} {new_end}" + "\t" + trimmed_text.replace("\n", " ") + "\n")

    assert isinstance(ID, str) and isinstance(folder, str)
    with open(os.path.join(folder, f"{ID}.ann"), mode, encoding='utf-8') as f:
        for item in to_write:
            f.write(item)

    return counter + len(to_write)

def evaluation_step(dev_dataloader: torch.utils.data.DataLoader,
                    model: torch.nn.Module,
                    device: torch.device,
                    columns: List[str],
                    idx2tag: Dict[int, str],
                    dev_str2num_file_names: Numericaliser,
                    pred_folder: str,
                    reference_data_folder: str,
                    text_folder_path: str,
                    post_process_type: str = "classic",
                    split_brat_on_newlines: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Evaluates a named entity recognition (NER) model using the BRATEval tool.

    Args:
        dev_dataloader: A DataLoader object containing the development dataset.
        model: The NER model to be evaluated.
        device: The device (CPU or GPU) to be used for inference.
        columns: A list of column names for the output.
        idx2tag: A dictionary mapping index to tag for the model predictions.
        dev_str2num_file_names: A Numericaliser of file names used for converting string to numerical values.
        pred_folder: The path to the prediction folder.
        reference_data_folder: The path to the reference data folder.
        text_folder_path: The path to the original text data folder.
        post_process_type: The type of post-processing to apply ('classic' or 'heuristic').
        split_brat_on_newlines: Whether to split annotations on newlines in BRAT format.

    Returns:
        A dictionary where keys are entity types and values are dictionaries containing the following metrics:
        TP: True Positives
        FP: False Positives
        FN: False Negatives
        Precision: Precision score
        Recall: Recall score
        F1: F1 score
    """
    # Get model predictions for the development dataset
    dev_predictions = infer(dev_dataloader,
                            model,
                            device,
                            columns,
                            idx2tag,
                            dev_str2num_file_names)

    # Convert sentence-level predictions into document-level predictions
    dev_doc_predictions = [sent2doc(i) for i in dev_predictions]

    # Loop through the document-level predictions
    counter = 0
    for c, dev_doc_prediction in enumerate(dev_doc_predictions):
        for pred in dev_doc_prediction:
            # Choose the post-process function based on the post_process_type
            assert post_process_type in ["heuristic", "classic"], "`post_process_type` should be either 'heuristic' or 'classic'."
            if post_process_type == "heuristic":
                post_process_func = post_process_heuristic
            else:
                post_process_func = post_process_classic

            # Post-process the predictions and get the annotation ID
            ann, ID = post_process_func(pred)

            # Write the annotations to the temporary folder, now with split_brat_on_newlines consideration
            counter = write_ann(ann, ID, text_folder_path, folder=pred_folder,
                                counter=counter, mode="a" if c > 0 else "w",
                                split_brat_on_newlines=split_brat_on_newlines)

    # Evaluate the model using the BRATEval tool and return the report
    report = evaluate(pred_folder, reference_data_folder)

    return report


def average_increase(x: List[float]) -> float:
    """
    This function takes a list of floats as an input and returns the average increase.
    If the input list has less than 2 elements, the function returns 0.

    Args:
    x : List[float]
        A list of floats

    Returns:
    float : Average increase
    """
    if len(x) < 2:
        return 0
    increases = [j-i for i, j in zip(x, x[1:])]
    average_increase = sum(increases) / len(increases)
    return average_increase


def convert_to_str(data: Dict) -> Dict:
    """
    This function converts all the values inside a dictionary to string.
    It uses recursion to iterate through the dictionary and convert all the values to strings.
    It maintains the original structure of the data.

    Parameters:
    - data (Dict) : The data that you want to convert the values to strings.

    Returns:
    - Dict : The data with all the values converted to strings
    """
    # check if data is a dictionary
    if isinstance(data, dict):
        # recursively iterate through the dictionary
        return {k: convert_to_str(v) for k, v in data.items()}
    # if not a dictionary, convert the value to string
    return str(data)


def save_to_json(data: Dict,
                 filename: str,
                 indent: Optional[int] = 4) -> None:
    """
    This function saves the passed data to a json file with the specified filename.
    It also uses the `indent` parameter to specify the number of spaces to use for indentation in the json file.

    Parameters:
    - data (Dict) : The data that you want to save in json format
    - filename (str) : The filename (including path) where you want to save the json file
    - indent (Optional[int]) : Number of spaces to use for indentation (default: 4)

    Returns:
    - None
    """
    # Open the file with write mode
    data_str = convert_to_str(data)
    with open(filename, 'w') as f:
        json.dump(data_str, f, indent=indent)