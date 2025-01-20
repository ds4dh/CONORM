import argparse
import os
import torch
from torch.nn import functional as F
from src.conorm_en_helpers import (
    load_config,
    build_en_dict_from_MedDRA,
    AutoTokenizer,
    AutoModel,
    CONORMEN,
    compute_vocabulary_embeddings,
    doc2sent_and_adjust_indices_spacy
)
from src.conorm_en_inference_helpers import *
from pprint import pprint
import json

def main() -> None:
    """
    This function infer from raw txt files.

    Returns:
        None
    """

    # PREPROCESSING

    print()
    print("Inferring from data...")

    parser = argparse.ArgumentParser(description='Infer from an EN model')
    parser.add_argument('config_path', type=str,
                        help='Path to the configuration file')

    args = parser.parse_args()
    
    config = load_config(args.config_path)

    # Accessing configuration variables
    path_to_txt = config["path_to_txt"]
    path_to_ann = config["path_to_ann"]
    path_to_model = config["path_to_model"]
    output_path = config["output_path"]
    inference_meddra_level = config["inference_meddra_level"]
    device = config["device"]
    batch_size = config["batch_size"]
    
    # Load used configuration
    used_config_path = os.path.join(path_to_model, "used_config.json")
    used_config = load_config(used_config_path)
    
    # Accessing used configuration variables during training
    llt_asc_path = used_config['llt_asc_path']
    pt_asc_path = used_config['pt_asc_path']
    meddra_level = used_config['meddra_level']
    isolated_model_name = used_config['isolated_model_name']
    context_model_name = used_config['context_model_name']
    document_level = used_config['document_level']

    # Save config in the parent folder of the output_path
    parent_folder = os.path.dirname(output_path)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    config_save_path = os.path.join(parent_folder, f"prediction_config_{meddra_level}_backbone.json")
    with open(config_save_path, 'w') as config_file:
        json.dump(config, config_file, indent=4)
    
    print()
    print("-"*15, " MODEL TRAINING DETAILS ", "-"*15)
    pprint(used_config)
    print()
    print("-"*15, " MODEL INFERENCE DETAILS ", "-"*15)
    pprint(config)
    print()
    
    # Sanity checks
    if inference_meddra_level not in ["llt", "pt"]:
        raise ValueError('inference_meddra_level should be either "llt" or "pt".')
    elif meddra_level == "llt":
        assert inference_meddra_level in ["llt", "pt"], "A model trained at the llt level can infer at the llt and pt level"
    elif meddra_level == "pt":
        assert inference_meddra_level in ["pt"], "A model trained at the pt level can only infer at the pt level"
    
    # Building dictionaries from MedDRA
    llt_dict, llt_to_pt, pt_dict, pt_to_pt = build_en_dict_from_MedDRA(llt_asc_path, pt_asc_path)
    
    # Selecting the appropriate dictionary based on MedDRA level
    if meddra_level == "llt":
        code_dict = llt_dict
        code_to_higher = llt_to_pt
    elif meddra_level == "pt":
        code_dict = pt_dict
        code_to_higher = pt_to_pt
    else:
        raise ValueError('meddra_level should be either "llt" or "pt".')
    
    inference_code_dict = llt_dict if inference_meddra_level == "llt" else pt_dict
    
    # Prepare data for inference
    infer_df = create_dataframe_from_ann_files_for_inference(path_to_ann, path_to_txt)
    if document_level:
        pass
    else:
        print()
        print("Tokenizing sentences from txt files... This may take some time.")
        print()
        infer_df[['text', 'start', 'end']] = infer_df.parallel_apply(
            lambda row: pd.Series(doc2sent_and_adjust_indices_spacy(row)), axis=1)
    
    # Model setup
    isolated_tokenizer = AutoTokenizer.from_pretrained(isolated_model_name)
    isolated_model = AutoModel.from_pretrained(isolated_model_name)
    
    context_tokenizer = AutoTokenizer.from_pretrained(context_model_name)
    context_tokenizer.add_tokens(["<START_ENTITY>", "<END_ENTITY>"])
    context_model = AutoModel.from_pretrained(context_model_name)
    context_model.resize_token_embeddings(len(context_tokenizer))
    
    model = CONORMEN(isolated_model=isolated_model, context_model=context_model)
    model.load_state_dict(torch.load(os.path.join(path_to_model, 'best_model.pt'), map_location=device))
    model.to(device)
    model.eval()
    
    # Create batches
    batches = create_encoded_inference_dataframe(infer_df, context_tokenizer, isolated_tokenizer, batch_size)
    
    # Compute vocabulary embeddings
    all_code_texts = list(code_dict.values())
    vocabulary_embeddings = compute_vocabulary_embeddings(isolated_model, isolated_tokenizer, all_code_texts, batch_size, device)
    
    # Inference
    outputs = []
    doc_ids, t_ids = [], []
    for batch in batches:
        doc_id, t_id, encoded_isolated, encoded_contexts = batch
        with torch.no_grad():
            attended_embeddings = model(encoded_isolated=encoded_isolated.to(device), encoded_contexts=encoded_contexts.to(device))
            attended_embeddings = F.normalize(attended_embeddings, p=2, dim=1)
        doc_ids.extend(doc_id)
        t_ids.extend(t_id)
        outputs.extend(attended_embeddings.to("cpu"))
    
    # Processing predictions
    outputs = torch.stack(outputs)
    target_vocab_emb = torch.stack(list(vocabulary_embeddings.values())).transpose(-1, -2)
    target_vocab_keys = list(vocabulary_embeddings.keys())
    pred_indices = (outputs @ target_vocab_emb).argmax(1)
    inv_code_dict = {v: k for k, v in code_dict.items()}
    text_predictions = [target_vocab_keys[i] for i in pred_indices]
    code_predictions = [inv_code_dict[text] for text in text_predictions]
    if inference_meddra_level == "pt":
        code_predictions = [code_to_higher[code] for code in code_predictions]
    
    # Merging predictions
    merged_dict = {}
    for doc_id, t_id, code in zip(doc_ids, t_ids, code_predictions):
        if doc_id not in merged_dict:
            merged_dict[doc_id] = []
        merged_dict[doc_id].append([t_id, code])
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Write .ann predictions
    # Iterate through documents and their predictions
    for doc_id, predictions in merged_dict.items():
        # Construct the path to read the annotation file for the current document
        path_to_read_doc_ann = os.path.join(path_to_ann, doc_id)
        # Read the existing annotations from the file
        try:
            with open(path_to_read_doc_ann, 'r') as file:
                doc_ann = file.read()
        except FileNotFoundError:
            print(f"Warning: File {path_to_read_doc_ann} not found. Skipping document {doc_id}.")
            continue  # Skip this document if the annotation file does not exist
        # Process each prediction for the current document
        for counter, prediction in enumerate(predictions):
            T_ID, CODE = prediction
            # Attempt to fetch the textual representation for the given code
            try:
                CODE_TEXT = inference_code_dict[CODE]
            except KeyError:
                # Handle missing codes gracefully
                CODE_TEXT = "TEXT_ERROR"
                print(f"\nFor document {doc_id}:")
                print(f"Couldn't fetch the term for code {CODE} in {inference_meddra_level}.")
                print("The prediction will still be written in the .ann file but there will be no text representation.")
                print("This error might be due to a discrepancy between the llt.asc and pt.asc files used to train the model.")
            # Prepare the string to append to the annotation
            to_append = f"N{counter+1}\tReference {T_ID} meddra_{inference_meddra_level}_id:{CODE}\t{CODE_TEXT}\n"
            doc_ann += to_append
        # Construct the path to write the updated annotation file
        path_to_write_doc_ann = os.path.join(output_path, doc_id)
        # Write the updated annotations back to the file
        with open(path_to_write_doc_ann, 'w') as file:
            file.write(doc_ann)


if __name__ == "__main__":
    main()