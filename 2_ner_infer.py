from src.inference_helpers import (
    save_dataset,
    create_dataset,
    read_json_file,
    merge_sentences,
    load_pickle,
    load_data,
    align_data,
    create_dataloader,
    MyAutoModelForTokenClassification,
    infer,
    sent2doc,
    post_process_classic,
    post_process_heuristic,
    most_frequent,
    write_ann
)
import torch
import argparse
import json
import os
import shutil
from transformers import AutoTokenizer
import logging
import shutil
from pprint import pprint


def main() -> None:
    """
    This function infer from raw txt files.

    Returns:
        None
    """

    # PREPROCESSING

    print()
    print("Inferring from data...")

    parser = argparse.ArgumentParser(description='Infer from a NER model')
    parser.add_argument('config_path', type=str,
                        help='Path to the configuration file')

    args = parser.parse_args()

    # Read configuration settings
    with open(args.config_path, 'r') as config_file:
        config = json.load(config_file)

    path_to_txt = config['path_to_txt']
    path_to_model = config['path_to_model']
    output_path = config['output_path']
    device = config['device']
    batch_size = config['batch_size']

    used_config = read_json_file(
        os.path.join(path_to_model, 'used_config.json'))
    print()
    print("-"*15, " MODEL TRAINING DETAILS ", "-"*15)
    pprint(used_config)
    print()
    print("-"*15, " MODEL INFERENCE DETAILS ", "-"*15)
    pprint(config)
    print()

    # Create BIO dataset using the `create_dataset` function from inference_helpers
    infer_bio_dataset = create_dataset(
        path_to_txt, os.path.join(path_to_model, 'tokenizer.pickle'))

    # Save the BIO datasets to the file system
    bio_saving_path = os.path.join(output_path, 'infer_bio')
    if not os.path.exists(bio_saving_path):
        os.makedirs(bio_saving_path)

    save_dataset(infer_bio_dataset, os.path.join(bio_saving_path, 'infer.txt'))

    if used_config['document_level']:
        merge_sentences(bio_saving_path)

    # INFERENCE

    text, labels, starts, ends, file_names, _ = load_data(
        os.path.join(output_path, 'infer_bio'))
    tag2idx = read_json_file(os.path.join(path_to_model, 'tag2idx.json'))
    columns = load_pickle(os.path.join(path_to_model, 'columns.pickle'))

    tokenizer = AutoTokenizer.from_pretrained(used_config['pretrained_model'])

    print()
    aligned_text, aligned_labels, aligned_subword_dummies, aligned_starts, aligned_ends, aligned_file_names = align_data(['infer'],
                                                                                                                         text,
                                                                                                                         labels,
                                                                                                                         starts,
                                                                                                                         ends,
                                                                                                                         file_names,
                                                                                                                         tokenizer,
                                                                                                                         used_config['max_len'])

    print()
    infer_dataloader, infer_str2num_file_names = create_dataloader("infer",
                                                                   aligned_text,
                                                                   aligned_labels,
                                                                   aligned_subword_dummies,
                                                                   aligned_starts,
                                                                   aligned_ends,
                                                                   aligned_file_names,
                                                                   tokenizer,
                                                                   tag2idx,
                                                                   used_config['max_len'],
                                                                   batch_size)

    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    model = MyAutoModelForTokenClassification(pretrained_model_name_or_path=used_config['pretrained_model'],
                                              tag2idx=tag2idx,
                                              columns=columns)

    model.load_state_dict(torch.load(
        os.path.join(path_to_model, 'best_model.pt'), map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        predictions = infer(infer_dataloader,
                            model,
                            device,
                            columns,
                            tag2idx,
                            infer_str2num_file_names)

    # Convert sentence-level predictions into document-level predictions
    doc_predictions = [sent2doc(i) for i in predictions]

    # Loop through the document-level predictions
    counter = 0
    for c, doc_prediction in enumerate(doc_predictions):
        for pred in doc_prediction:

            # Choose the post-process function based on the post_process_type
            post_process_type = used_config["post_process_type"]
            assert post_process_type in [
                "heuristic", "classic"], "`post_process_type` should be either 'heuristic' or 'classic'."
            if post_process_type == "heuristic":
                post_process_func = post_process_heuristic
            else:
                post_process_func = post_process_classic

            # Post-process the predictions and get the annotation ID
            ann, ID = post_process_func(pred)

            # Write the annotations to the temporary folder
            counter = write_ann(ann, ID, path_to_txt, folder=output_path, counter=counter, mode="a" if c >
                                0 else "w", split_brat_on_newlines=used_config["split_brat_on_newlines"])

    shutil.rmtree(bio_saving_path)


if __name__ == "__main__":
    main()