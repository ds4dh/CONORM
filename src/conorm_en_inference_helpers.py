import os
import pandas as pd
from tqdm.auto import tqdm

def create_dataframe_from_ann_files_for_inference(folder_path, txt_folder_path):
    """
    Create a pandas DataFrame from .ann files in the given folder for inference purposes,
    using separate paths for .ann and corresponding .txt files.

    Parameters:
    - folder_path: The path to the folder containing the .ann files.
    - txt_folder_path: The path to the folder containing the corresponding .txt files.
    """
    all_info = []
    for file_name in tqdm(os.listdir(folder_path), desc="Parsing .ann files for inference"):
        if file_name.endswith('.ann'):
            ann_path = os.path.join(folder_path, file_name)
            
            # Construct the .txt file path based on the .ann file name
            txt_file_name = file_name.replace('.ann', '.txt')
            txt_file_path = os.path.join(txt_folder_path, txt_file_name)
            
            # Parse the .ann file for "T" row information
            info = parse_ann_for_inference(ann_path, txt_file_path)
            all_info.extend(info)

    df = pd.DataFrame(all_info)
    return df

def parse_ann_for_inference(ann_path, txt_file_path):
    """
    Parse an .ann file to extract information from "T" rows for inference,
    using the corresponding .txt file for the full text.

    Parameters:
    - ann_path: Path to the .ann file.
    - txt_file_path: Path to the .txt file containing the full document text.
    """
    t_info = []  # List to store extracted information from T rows

    with open(txt_file_path, 'r') as txt_file:
        full_text = txt_file.read()

    with open(ann_path, 'r') as file:
        for line in file:
            if line.startswith('T'):
                parts = line.strip().split('\t')
                t_id = parts[0]
                entity_info = parts[1].split(' ')
                start, end = int(entity_info[1]), int(entity_info[2])
                
                t_info.append({
                    'doc_id': ann_path.split("/")[-1],  # Corrected by adding a comma
                    't_id': t_id,
                    'start': start,
                    'end': end,
                    'text': full_text
                })

    return t_info

def annotate_concepts(input_text, start_indices, end_indices):
    """
    Annotate the input text with <START_ENTITY> and <END_ENTITY> tokens.

    Args:
        input_text (List[str]): Input texts.
        start_indices (List[int]): Start indices of the concepts in the texts.
        end_indices (List[int]): End indices of the concepts in the texts.

    Returns:
        List[str]: Annotated texts.
    """
    return [f"{text[:start]}<START_ENTITY>{text[start:end]}<END_ENTITY>{text[end:]}"
            for text, start, end in zip(input_text, start_indices, end_indices)]

def create_encoded_inference_dataframe(df, context_tokenizer, isolated_tokenizer, batch_size):

    TEXT_MAX_LEN = 512 if context_tokenizer.model_max_length > 99999 else None
    CONCEPT_MAX_LEN = 512 if isolated_tokenizer.model_max_length > 99999 else None
    
    # Initialize batches
    doc_ids, t_ids, starts, ends, texts = [], [], [], [], []
    batches = []

    for instance in df[["doc_id", "t_id", "start", "end", "text"]].values.tolist():
        doc_id, t_id, start, end, text = instance
        
        # Collect instances for a batch
        doc_ids.append(doc_id)
        t_ids.append(t_id)
        starts.append(start)
        ends.append(end)
        texts.append(text)
        
        # Check if batch is full or if it's the last instance
        if len(starts) == batch_size or instance == df[["doc_id", "t_id", "start", "end", "text"]].values.tolist()[-1]:
            
            # Process Context
            isolated = [t[s:e] for t, s, e in zip(texts, starts, ends)]
            encoded_isolated = isolated_tokenizer(isolated, padding=True, truncation=True, return_tensors='pt', max_length=CONCEPT_MAX_LEN)
            
            # Process Text
            annotated_contexts = annotate_concepts(texts, starts, ends)
            encoded_contexts = context_tokenizer(annotated_contexts, padding=True, truncation=True, return_tensors='pt', max_length=TEXT_MAX_LEN)
            
            # Append to batches
            batches.append((doc_ids, t_ids, encoded_isolated, encoded_contexts))
            
            # Reset for next batch
            doc_ids, t_ids, starts, ends, texts = [], [], [], [], []
    
    return batches