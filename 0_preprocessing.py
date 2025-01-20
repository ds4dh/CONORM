from src.standoff2bio import train_and_save_tokenizer, create_dataset, merge_sentences
import os
import json
import logging
from typing import List
import argparse
import shutil
import sys
import nltk
import pickle


def extract_unique_labels(dataset: str) -> set:
    """Extract unique labels from the given dataset."""
    labels = set()
    lines = dataset.strip().split('\n')
    
    for line in lines:
        if line:
            tokens = line.split('\t')[1:-3]  # Excluding token, start, end, and filename columns
            for label in tokens:
                label_name = label.split('-')[-1]
                labels.add(label_name)

    return labels

def harmonize_dataset_columns(dataset: str, sorted_all_labels: list) -> str:
    """Harmonize the dataset columns by adding and ordering labels."""
    lines = dataset.strip().split('\n')
    harmonized_data = []

    for line in lines:
        if line:  # Check if line is not empty
            tokens = line.split('\t')
            current_token = tokens[0]
            current_labels_dict = {label: 'O-' + label for label in sorted_all_labels}  # Initialize all labels with default 'O-' tag
            
            existing_labels = tokens[1:-3]  # Excluding token, start, end, and filename columns
            for label in existing_labels:
                current_labels_dict[label.split('-')[-1]] = label  # Update the label with the correct B- or I- tag if present

            harmonized_labels = [current_labels_dict[label] for label in sorted_all_labels]
            
            harmonized_line = current_token + '\t' + '\t'.join(harmonized_labels) + '\t' + '\t'.join(tokens[-3:])
            harmonized_data.append(harmonized_line)
        else:
            harmonized_data.append('')  # Preserve the empty lines separating sentences

    return '\n'.join(harmonized_data)


def save_dataset(data: str, path: str) -> None:
    """
    Save datasets as text files.

    Args:
        data (str): The training dataset.
        path (str): The path to save the datasets.

    Returns:
        None
    """
    # Save the training dataset
    with open(path, 'w') as f:
        f.write(data)


def main() -> None:
    """
    This function creates simple and BIO datasets, and saves them to the file system.

    Returns:
        None
    """

    print()
    print("Preprocessing data...")
    print()

    parser = argparse.ArgumentParser(description='Create a formatted dataset for the NER pipeline')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')

    args = parser.parse_args()

    # Read configuration settings
    with open(args.config_path, 'r') as config_file:
        config = json.load(config_file)
    print("config loaded.\n")

    train_path = config['train_standoff_path']
    val_path = config['val_standoff_path']
    test_path = config.get('test_standoff_path')
    document_level = config['document_level']
    
    # Fallback to val_path if test_path is not provided or is None
    if test_path is None:
        test_path = os.path.join(os.path.dirname(val_path), 'test')

        if os.path.exists(test_path):
            logging.error("Test directory already exists but test_standoff_path is None. Stopping script.")
            sys.exit("Error: Test directory already exists but test_standoff_path is None.")

        os.makedirs(test_path)
        logging.info(f"Created synthetic test directory: {test_path}")

        # Verify and copy files from val_path
        if os.path.isdir(val_path):
            files_in_val = os.listdir(val_path)

            if not files_in_val:
                logging.warning("No files found in val directory to copy into synthetic test folder.")
                sys.exit("Error: No files found in val directory to copy into synthetic test folder.")
            else:
                logging.warning("Copying validation files into synthetic test folder.")

            for file_name in files_in_val:
                full_file_name = os.path.join(val_path, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, test_path)
        else:
            logging.error(f"val_path is not a directory: {val_path}")
    
    # Create BIO datasets using the `create_dataset` function
    if config['train_sentence_tokenizer']:
        train_and_save_tokenizer(train_path)
        tokenizer_path = os.path.join(os.path.dirname(train_path), 'tokenizer.pickle')
    else:
        # Load the default Punkt tokenizer
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        
        # Save the tokenizer to the new location
        tokenizer_path = os.path.join(os.path.dirname(train_path), 'tokenizer.pickle')
        with open(tokenizer_path, 'wb') as file:
            pickle.dump(tokenizer, file)
    
    train_bio_dataset = create_dataset(train_path, tokenizer_path)
    val_bio_dataset = create_dataset(val_path, tokenizer_path)
    test_bio_dataset = create_dataset(test_path, tokenizer_path)

    # Find all unique labels across the train, validation, and test datasets
    all_labels = set()
    all_labels.update(extract_unique_labels(train_bio_dataset))
    all_labels.update(extract_unique_labels(val_bio_dataset))
    all_labels.update(extract_unique_labels(test_bio_dataset))

    # Convert the set of labels into a sorted list
    sorted_all_labels = sorted(all_labels)

    # Check if the datasets need harmonization
    if set(sorted_all_labels) != set(extract_unique_labels(train_bio_dataset)) or \
       set(sorted_all_labels) != set(extract_unique_labels(val_bio_dataset)) or \
       set(sorted_all_labels) != set(extract_unique_labels(test_bio_dataset)):
        logging.warning("Datasets will be harmonized due to inconsistent labels across splits.")
    else:
        logging.warning("Datasets do not need harmonization.")

    # Harmonize the columns for each dataset
    train_bio_dataset = harmonize_dataset_columns(train_bio_dataset, sorted_all_labels)
    val_bio_dataset = harmonize_dataset_columns(val_bio_dataset, sorted_all_labels)
    test_bio_dataset = harmonize_dataset_columns(test_bio_dataset, sorted_all_labels)

    # Save the BIO datasets to the file system
    bio_saving_path = os.path.join(os.path.dirname(train_path), 'bio')
    if not os.path.exists(bio_saving_path):
        os.makedirs(bio_saving_path)
        
    save_dataset(train_bio_dataset, os.path.join(bio_saving_path, 'train.txt'))
    save_dataset(val_bio_dataset, os.path.join(bio_saving_path, 'val.txt'))
    save_dataset(test_bio_dataset, os.path.join(bio_saving_path, 'test.txt'))

    if document_level:
        merge_sentences(bio_saving_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
