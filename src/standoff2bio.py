import re
import os
import glob
from itertools import chain
from tqdm import tqdm
import collections
from typing import Tuple, Dict, List
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import pickle
import regex

def train_and_save_tokenizer(folder_path):
    def load_text_files(folder_path):
        text_data = ""
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as file:
                    text_data += file.read() + "\n\n"
        return text_data

    def train_punkt_tokenizer(text):
        trainer = PunktTrainer()
        #trainer.INCLUDE_ALL_COLLOCS = True
        trainer.train(text)
        return PunktSentenceTokenizer(trainer.get_params())

    def save_tokenizer(tokenizer, filename):
        with open(filename, 'wb') as file:
            pickle.dump(tokenizer, file)

    text = load_text_files(folder_path)
    tokenizer = train_punkt_tokenizer(text)

    save_file = os.path.join(os.path.dirname(folder_path), 'tokenizer.pickle')
    save_tokenizer(tokenizer, save_file)


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
    files_paths = glob.glob(path_to_data + '/*.txt')

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

    files_annotations_paths = [os.path.splitext(i)[0] + ".ann" for i in files_paths]

    # Initialize an empty dictionary to store the file names and contents
    files_annotations = {}

    # Iterate over the list of file paths
    for file_annotations_path in files_annotations_paths:
        # Open the file and read its contents
        with open(file_annotations_path, 'r') as f:
            # Extract the file name from the file path
            file_name = os.path.splitext(
                os.path.basename(file_annotations_path))[0]

            # Add the file name and content to the dictionary
            files_annotations[file_name] = f.read()

    return files, files_annotations


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


def process_annotations(files: dict, files_annotations: dict, tokenized_files: dict, tokens_indexes: dict, BIO: bool) -> str:
    """
    Processes the annotations in the given dictionaries and returns the annotated dataset in BIO format or with the annotation labels directly.

    Parameters
    ----------
    files : dict
        A dictionary where the keys are file names and the values are file contents.
    files_annotations : dict
        A dictionary where the keys are file names and the values are the annotation files corresponding to the keys in the `files` dictionary.
    tokenized_files : dict
        A dictionary where the keys are file names and the values are lists of lists of tokens, where each list of tokens represents a sentence and each token is a string.
    tokens_indexes : dict
        A dictionary where the keys are file names and the values are lists of lists of tuples, where each tuple represents the start and end indexes of a token in the corresponding file.
    BIO : bool
        A boolean indicating whether to use the BIO format (True) or the annotation labels directly (False).

    Returns
    -------
    str
        The annotated dataset in the specified format.
    """
    # Get unique anns
    unique_ann = []
    for file in files_annotations.values():
        for line in file.strip().split('\n'):
            if line and line[0] == 'T':
                line_split = line.split('\t')
                ann_number, ann, text = line_split[:3]
                ann_split = ann.split(' ')
                ann, start, end = ann_split[0], ann_split[1], ann_split[-1]
                unique_ann.append(ann)
    unique_ann = sorted(set(unique_ann))

    datasets = {ANN: None for ANN in unique_ann}

    for ANN in tqdm(unique_ann, leave=False):
        # Create a dictionary to store the processed annotations
        annotations = {}

        # Iterate over the keys in the files dictionary (which are the file names)
        for file_name in files.keys():
            # Initialize an empty dictionary for the current file
            annotations[file_name] = {}

            # Split the annotation file into lines and process each line
            for line in files_annotations[file_name].strip().split('\n'):
                if line and line[0] == 'T':
                    # Extract the annotation ID, annotation group, and text from the line
                    line_split = line.split('\t')
                    ann_number, ann, text = line_split[:3]
                    ann_split = ann.split(' ')
                    ann, start, end = ann_split[0], ann_split[1], ann_split[-1]

                    if ann not in annotations[file_name]:
                        annotations[file_name][ann] = {'range': [], 'text': []}
                    annotations[file_name][ann]['range'].append(
                        set([i for i in range(int(start), int(end))]))
                    annotations[file_name][ann]['text'].append(text)

        # Initialize an empty string to store the annotated dataset
        dataset = ''

        # Iterate over the tokenized text files
        for file_name in tokenized_files:
            prev_ann = f'O-{ANN}'
            prev_set = set()
            for i, sentence in enumerate(tokenized_files[file_name]):
                for j, token in enumerate(sentence):
                    start, end = tokens_indexes[file_name][i][j]
                    annotated = False
                    if ANN in annotations[file_name].keys():
                        curr_ranges = annotations[file_name][ANN]['range']
                        intersect_ranges = [r for r in curr_ranges if any(k in r for k in range(start, end))]
                        if intersect_ranges:
                            index_set = set.union(*intersect_ranges)
                            label_prefix = 'I-' if BIO and prev_ann == ANN and index_set == prev_set else 'B-'
                            dataset += token + '\t' + label_prefix + ANN + '\t' + \
                                       str(start) + '\t' + str(end) + \
                                       '\t' + file_name + '\n'
                            prev_ann = ANN
                            prev_set = index_set
                            annotated = True
                    if not annotated:
                        dataset += token + '\t' + f'O-{ANN}' + '\t' + \
                                   str(start) + '\t' + str(end) + \
                                   '\t' + file_name + '\n'
                        prev_ann = f'O-{ANN}'
                        prev_set = set()
                dataset += '\n'

        datasets[ANN] = dataset

    data_list = [i.split("\n") for i in datasets.values()]
    data_list = list(map(list, zip(*data_list)))
    output_list = []
    for item in data_list:
        if item[0]:
            fields = item[0].split("\t")
            tags = [j.split("\t")[1] for j in item]
            output_list.append(
                f"{fields[0]}\t" + "\t".join(tags) + f"\t{fields[2]}\t{fields[3]}\t{fields[4]}")
        else:
            output_list.append("")
    output = "\n".join(output_list)

    return output


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
    files, annotation_files = read_files(path_to_data)

    # Extract the whitespace indexes and tokens from the files
    whitespaces_indexes = extract_whitespace_indexes(files)
    tokenized_files, tokens_indexes = extract_tokens_and_indexes(files, whitespaces_indexes, tokenizer_path)

    # Process the annotations for the dataset
    dataset = process_annotations(files, annotation_files, tokenized_files, tokens_indexes, True)

    return dataset
    

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