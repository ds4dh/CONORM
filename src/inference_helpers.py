import os

# Save the original value of TF_CPP_MIN_LOG_LEVEL
original_tf_cpp_min_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL')

# Set TF_CPP_MIN_LOG_LEVEL to 3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import glob
from itertools import chain
from tqdm.auto import tqdm
from typing import Tuple
import nltk
import regex
import json
from typing import Dict, List, Union, Any, Optional
import pickle
import logging
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
from transformers import AutoModelForTokenClassification
from collections import defaultdict, Counter

# Set TF_CPP_MIN_LOG_LEVEL back to its original value
if original_tf_cpp_min_log_level is not None:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_tf_cpp_min_log_level
else:
    del os.environ['TF_CPP_MIN_LOG_LEVEL']

def replace_invisible_characters(text):
    # Regular expression pattern to match invisible characters
    pattern = r'[^\p{L}\p{N}\p{P}\p{Z}\p{Cf}\s]'

    # Replace matching characters with a white space
    return regex.sub(pattern, ' ', text)


def read_files(path_to_data: str) -> Tuple[dict, dict]:
    """
    Reads the text and annotation files in the specified path and stores the file names and content in a dictionary.

    Args:
        path_to_data: The path to the directory where the annotation files are located.

    Returns:
        A tuple of dictionaries with file names as keys and file contents as values.
    """
    # Get the list of file paths for all text files in the specified path
    files_paths = glob.glob(os.path.join(path_to_data, '*.txt'))

    # Initialize an empty dictionary to store the file names and contents
    files = {}

    # Iterate over the list of file paths
    for file_path in files_paths:
        # Open the file and read its contents
        with open(file_path, 'r') as f:
            # Extract the file name from the file path
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            # Read the file contents and replace double quotes with single quotes
            text = f.read().replace('"', "'")

            # Add the file name and content to the dictionary
            files[file_name] = replace_invisible_characters(text)

    return files


def extract_whitespace_indexes(files: dict) -> dict:
    """Extracts the indexes of whitespaces in the given files.

    Args:
        files: A dictionary mapping file names to file contents.

    Returns:
        A dictionary mapping file names to a list of integers representing the 
        indexes of whitespaces in the corresponding file content.
    """

    # Compile the regex pattern for matching whitespaces
    whitespace_pattern = re.compile(r'\s')

    # Initialize the dictionary to store the whitespace indexes
    whitespaces_indexes = {}

    # Iterate over the files and extract the whitespace indexes
    for file_name, file_content in files.items():
        whitespaces_indexes[file_name] = [
            m.start() for m in re.finditer(whitespace_pattern, file_content)]

    # Return the dictionary of whitespace indexes
    return whitespaces_indexes


def extract_tokens_and_indexes(input_files: dict, whitespaces_indexes: dict, tokenizer_path: str) -> tuple:
    """
    Tokenizes the input text and extracts the token indexes for each file.

    Args:
        input_files: A dictionary with file names as keys and file contents as values.
        whitespaces_indexes: A dictionary with file names as keys and lists of whitespace indexes as values.
        tokenizer_path: Path to the tokenizer.

    Returns:
        A tuple containing:
            - A dictionary with file names as keys and tokenized sentences as values.
            - A dictionary with file names as keys and lists of token indexes as values.
    """

    # Load Punkt tokenizer
    tokenizer = nltk.data.load(tokenizer_path)
    word_tokenizer = nltk.RegexpTokenizer(r'\b\w+\b|[^\s\w](?=[^\s\w])|[^\s\w]+')
    # Initialize empty dictionaries for storing tokenized files and token indexes
    tokenized_files = {}
    token_indexes = {}

    # Iterate over each file
    for file_name in tqdm(input_files.keys(), leave=False):
        # Initialize empty list for storing tokenized sentences and token indexes for this file
        tokenized_sentences = []
        token_indexes[file_name] = []
        # Tokenize sentences in file
        for sentence in tokenizer.tokenize(input_files[file_name]):
            # Tokenize words in sentence
            tokenized_sentences.append(word_tokenizer.tokenize(sentence))
        # Store tokenized sentences for this file
        tokenized_files[file_name] = tokenized_sentences
        # Flatten list of tokenized sentences
        flat_list = list(chain(*tokenized_sentences))
        # Initialize index_b to 0
        index_b = 0
        # Iterate over tokens
        for token in flat_list:
            # Append start and end index of token to list of token indexes for this file
            token_indexes[file_name].append([index_b, index_b+len(token)])
            # Increment index_b by length of token
            index_b += len(token)
        # Iterate over whitespace indexes for this file
        for whitespace_index in whitespaces_indexes[file_name]:
            # Iterate over token indexes for this file
            for token_index in token_indexes[file_name]:
                # If start index of token is greater than or equal to whitespace index, increment start and end index of token by 1
                if token_index[0] >= whitespace_index:
                    token_index[0] += 1
                    token_index[1] += 1
        # Assert that the number of token indexes is equal to the number of tokens
        assert len(token_indexes[file_name]) == len(flat_list)
        # Join list of tokens into a single string
        joined_list = ''.join(flat_list)
        # Iterate over whitespace indexes for this file
        for index in whitespaces_indexes[file_name]:
            # Insert a whitespace at each whitespace index
            joined_list = joined_list[:index] + ' ' + joined_list[index:]
        # Assert that the length of the joined list is equal to the length of the original content of the file
        assert len(input_files[file_name]) == len(joined_list)
        # Iterate over tokens and token indexes
        for i in range(len(token_indexes[file_name])):
            # Temporary variable for storing token index
            tmp = token_indexes[file_name][i]
            # Assert that the token is equal to the substring of the original content of the file at the start and end indexes of the token
            assert flat_list[i] == input_files[file_name][tmp[0]:tmp[1]]
        # Initialize empty list for storing token indexes by sentence
        token_indexes_by_sentence = []
        # Initialize index to 0
        i = 0
        # Iterate over tokenized sentences
        for sentence in tokenized_sentences:
            # Append an empty list to token_indexes_by_sentence
            token_indexes_by_sentence.append([])
            # Iterate over tokens in sentence
            for _ in sentence:
                # Append token index to inner list
                token_indexes_by_sentence[-1].append(
                    token_indexes[file_name][i])
                # Increment index by 1
                i += 1
        # Replace list of token indexes for this file with list of token indexes by sentence
        token_indexes[file_name] = token_indexes_by_sentence

    # Return tokenized files and token indexes
    return tokenized_files, token_indexes


def process_annotations(files: dict, tokenized_files: dict, tokens_indexes: dict) -> str:
    """
    Processes the tokens and assigns "O-INFERENCE" to each token, as there are no specific annotations.

    Parameters
    ----------
    files : dict
        A dictionary where the keys are file names and the values are file contents.
    tokenized_files : dict
        A dictionary where the keys are file names and the values are lists of lists of tokens, where each list of tokens represents a sentence and each token is a string.
    tokens_indexes : dict
        A dictionary where the keys are file names and the values are lists of lists of tuples, where each tuple represents the start and end indexes of a token in the corresponding file.

    Returns
    -------
    str
        The dataset with every token labeled as "O-INFERENCE".
    """
    # Initialize an empty string to store the annotated dataset
    dataset = ''

    # Iterate over the tokenized text files
    for file_name in tokenized_files:
        for i, sentence in enumerate(tokenized_files[file_name]):
            for j, token in enumerate(sentence):
                start, end = tokens_indexes[file_name][i][j]
                dataset += token + '\t' + 'O-INFERENCE' + '\t' + \
                           str(start) + '\t' + str(end) + '\t' + file_name + '\n'
            dataset += '\n'

    return dataset


def create_dataset(path_to_data: str, tokenizer_path: str) -> str:
    """
    Creates a dataset from the given data files.

    Parameters
    ----------
    path_to_data : str
        The path to the directory containing the data files.
    tokenizer_path: str
        Path to the tokenizer.

    Returns
    -------
    str
        The dataset as a string in the specified format.
    """
    # Read the text and annotation files and store them in dictionaries
    files = read_files(path_to_data)

    # Extract the whitespace indexes and tokens from the files
    whitespaces_indexes = extract_whitespace_indexes(files)
    tokenized_files, tokens_indexes = extract_tokens_and_indexes(files, whitespaces_indexes, tokenizer_path)

    # Process the annotations for the dataset
    dataset = process_annotations(files, tokenized_files, tokens_indexes)

    return dataset

def save_dataset(data: str, path: str) -> None:
    """
    Save dataset as text files.

    Args:
        data (str): The dataset.

    Returns:
        None
    """
    # Save the training dataset
    with open(path, 'w') as f:
        f.write(data)


def merge_sentences(path: str) -> None:
    """
    Merge instances of the same document together in a text file based on their ID in the 
    last column of each line. A blank line is inserted between different documents. The 
    resulting documents overwrite the original files.

    It is used to go from sentence level modeling to document level modeling.

    Args:
        path (str): The path to the directory containing the text files.
    """
    # Ensure path exists
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return

    # Iterate over all files in the specified directory
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)

            # Read the data from the file
            with open(file_path, 'r') as file:
                data = file.readlines()

            # Initialize storage for new file content and previous document ID
            new_data = []
            prev_doc_id = None

            for line in data:
                if line.strip() == '':  # Skip if line is empty
                    continue

                # Extract document ID from the last column
                doc_id = line.split('\t')[-1].strip()

                # If this is a new document, add a newline to separate from previous document
                if doc_id != prev_doc_id and prev_doc_id is not None:
                    new_data.append('\n')

                # Add line to new data
                new_data.append(line)

                # Update previous document ID
                prev_doc_id = doc_id

            # Write the new data back to the file
            with open(file_path, 'w') as file:
                file.writelines(new_data)


def read_json_file(file_path):
    """
    Load json file from path.
    """
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return None

def load_pickle(filename: str) -> List:
    """
    Load a pickle file.

    Args:
        filename (str): The name of the file to load the pickle from.
        
    Raises:
        IOError: If an error occurs while reading the file.
        
    Returns:
        List: The loaded pickle.
    """
    try:
        with open(filename, 'rb') as file:
            my_list = pickle.load(file)
        return my_list
    except IOError as e:
        print(f"An error occurred while loading the list: {e}")


def load_data(data_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]], 
                                      Dict[str, List[List[int]]], Dict[str, List[List[int]]], 
                                      Dict[str, List[List[str]]], List[str]]:
    """
    Load the data from the specified files, assuming the data group is always 'infer'.

    Args:
        data_dir: Directory where the data files are stored.

    Returns:
        Tuple of dictionaries with sentences and labels.
    """
    data_group = 'infer'
    data_path = data_dir + "/" + data_group + '.txt'

    sentences = {data_group: []}
    starts = {data_group: []}
    ends = {data_group: []}
    file_names = {data_group: []}
    labels = {data_group: []}

    with open(data_path, 'r', encoding='utf-8') as f:
        file = f.read()

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

    logging.warning(" " + data_group + ' number of instances = ' + str(len(sentences[data_group])))

    tags_vals = list(chain(*labels[data_group]))
    transposed_labels = zip(*tags_vals)
    unique_label_per_column = []
    for column in transposed_labels:
        column = [i.split("-")[-1] for i in column]
        unique_label = [x for x in list(set(column)) if x != 'O'][0]
        unique_label_per_column.append(unique_label)

    return sentences, labels, starts, ends, file_names, unique_label_per_column


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
                tags_ = np.array([[t if i == 0 else f"X-{t.split('-')[-1]}" for i, _ in enumerate(sub_words)] for t in tag])

                BERT_subwords_dummy.extend([0] + [1] * (len(sub_words) - 1))
                BERT_texts.extend(sub_words)
                BERT_labels = tags_ if BERT_labels.size == 0 else np.hstack([BERT_labels, tags_])
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
        logging.warning(f' Too long instances in {data_group} = {too_big_count} (truncated to {MAX_LEN} tokens)')
        logging.warning(f' Misaligned text/label instances in {data_group} = {missalign_count}')

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
                      BATCH_SIZE: int) -> Tuple[DataLoader, Numericaliser, Numericaliser]:
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
            [pad_sequences([[0 for i in l] for l in lab],
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

        seq = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(seq.last_hidden_state)
        seq = [seq for layer in self.classification_layers]
        seq = [torch.cat((sub_seq, subword_dummies.unsqueeze(-1)), dim=-1) for sub_seq in seq]
        logits = [layer.classifier(sub_seq) for layer, sub_seq in zip(self.classification_layers, seq)]

        if self.training:
            flat_mask = attention_mask.view(-1)
            flat_logits = [logits[i].view(-1, self.num_labels)[flat_mask != 0, :] for i in range(len(logits))]
            flat_labels = [labels[:, :, i].view(-1)[flat_mask != 0] for i in range(len(logits))]
            ce_loss = torch.mean(torch.stack([
                layer.ce(flat_logits[i], flat_labels[i])
                for i, layer in enumerate(self.classification_layers)]))
            return (ce_loss, None)
        else:
            if labels is not None:
                flat_mask = attention_mask.view(-1)
                flat_logits = [logits[i].view(-1, self.num_labels)[flat_mask != 0, :] for i in range(len(logits))]
                flat_labels = [labels[:, :, i].view(-1)[flat_mask != 0] for i in range(len(logits))]
                ce_loss = torch.mean(torch.stack([
                    layer.ce(flat_logits[i], flat_labels[i])
                    for i, layer in enumerate(self.classification_layers)]))
                return (ce_loss, logits)
            else:
                return (None, logits)


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
        file_names = filtered_true_sequence(subword_dummies, file_names, input_mask)

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