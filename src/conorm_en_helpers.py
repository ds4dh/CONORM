import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Tuple
import spacy
import re
from tqdm.auto import tqdm, trange
import pandas as pd
import os
import json
from pandarallel import pandarallel
from datetime import datetime

pandarallel.initialize(progress_bar=False)


def create_path(train_standoff_path: str,
                isolated_model_name: str,
                context_model_name: str) -> str:
    """
    Create a path for saving logs based on the inputs.

    Args:
        train_standoff_path (str): The folder path where train files are located.
        pretrained_model (str, optional): The pretrained model to use. Defaults to None.

    Returns:
        str: The path for saving the logs.
    """
    # Get the current date and time
    date = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Get the language data and entities
    lang_data = "_".join(train_standoff_path.split("/")[2:4])

    # Define model name
    model_name = isolated_model_name.replace(
        "/", "_") + "_" + context_model_name.replace("/", "_") + "_" + date
    model_name = model_name.replace("/", "_")

    # Define the path to save the logs
    PATH = os.path.join('.', 'logs', 'conormen', lang_data, model_name)

    return PATH


def load_config(file_path):
    """
    Load a JSON configuration file.

    Parameters:
    - file_path (str): Path to the JSON configuration file.

    Returns:
    - config (dict): A dictionary containing the configuration.
    """
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


def save_config(config, file_path):
    """
    Save a configuration dictionary to a JSON file.

    Parameters:
    - config (dict): Configuration to save.
    - file_path (str): Path where the JSON file will be saved.
    """
    with open(file_path, 'w') as file:
        json.dump(config, file, indent=4)


def build_en_dict_from_MedDRA(path2lltasc: str, path2ptasc: str):
    """
    Builds dictionaries from MedDRA llt and pt files, including a self-mapping pt_to_pt.
    """

    if not os.path.exists(path2ptasc):
        print("Error: Folder Not Found ", path2ptasc)
        return

    pt_dict = {}
    pt_to_hlt = {}
    pt_to_pt = {}  # Initialize the pt_to_pt dictionary

    with open(path2ptasc, "r", encoding="utf-8") as file:
        for line in file:
            fs = line.strip().split("$")
            pt = fs[0]
            text = fs[1]
            hlt = fs[2]

            if pt not in pt_dict:
                pt_dict[pt] = text
                pt_to_pt[pt] = pt  # Map pt to itself
            else:
                print("Duplicate PT code found: ", pt)

            pt_to_hlt[pt] = hlt

    if not os.path.exists(path2lltasc):
        print("Error: Folder Not Found ", path2lltasc)
        return

    llt_dict = {}
    llt_to_pt = {}

    with open(path2lltasc, "r", encoding="utf-8") as file:
        for line in file:
            fs = line.strip().split("$")
            llt = fs[0]
            text = fs[1]
            pt = fs[2]

            if llt not in llt_dict:
                llt_dict[llt] = text
            else:
                print("Duplicate LLT code found: ", llt)

            llt_to_pt[llt] = pt

    # Include pt_to_pt in the return statement
    return llt_dict, llt_to_pt, pt_dict, pt_to_pt


# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


def parse_ann_for_specific_rows(ann_path, parse_type, code_dict):
    """
    Parse an .ann file to extract information based on "N" rows for a specific parse_type (llt or pt),
    and their corresponding "T" rows. The 'text' column will contain the full text from the corresponding .txt file.

    Parameters:
    - ann_path: Path to the .ann file.
    - parse_type: Specifies which type of information to parse ('llt' or 'pt').
    """
    t_info = {}  # Store start and end positions keyed by T id
    specific_info = []  # Store extracted specific row information

    # Determine the path for the corresponding .txt file
    txt_path = ann_path.replace('.ann', '.txt')
    with open(txt_path, 'r') as txt_file:
        full_text = txt_file.read()

    with open(ann_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if line.startswith('T'):
                # Parsing T rows for start, end
                t_id = parts[0]
                entity_info = parts[1].split(' ')
                start, end = int(entity_info[1]), int(entity_info[2])
                t_info[t_id] = {'start': start, 'end': end}
            elif line.startswith('N') and (('meddra_' + parse_type + '_id') in parts[1]):
                # Parsing N rows for specific type (llt or pt)
                ref_id = re.search(r'Reference (T\d+)', parts[1]).group(1)
                code = re.search(r'meddra_' + parse_type +
                                 '_id:(\d+)', parts[1]).group(1)
                code_text = parts[2] if len(parts) > 2 else ""
                if ref_id in t_info:
                    if code not in code_dict:
                        continue
                    specific_info.append({
                        'start': t_info[ref_id]['start'],
                        'end': t_info[ref_id]['end'],
                        'code': code,
                        'code_text': code_text,
                        'text': full_text  # Use the full text from the .txt file
                    })

    return specific_info


def create_dataframe_from_ann_files(folder_path, parse_type, code_dict):
    """
    Create a pandas DataFrame from .ann files in the given folder, focusing on specific annotations (llt or pt).
    The 'text' column will contain the full text from the corresponding .txt file.

    Parameters:
    - folder_path: The path to the folder containing the .ann files.
    - parse_type: Specifies which type of information to parse ('llt' or 'pt').
    """
    all_info = []
    for file_name in tqdm(os.listdir(folder_path), leave=False, desc="Creating formatted sets"):
        if file_name.endswith('.ann'):
            ann_path = os.path.join(folder_path, file_name)
            info = parse_ann_for_specific_rows(
                ann_path, parse_type=parse_type, code_dict=code_dict)
            all_info.extend(info)

    df = pd.DataFrame(all_info, columns=[
                      'start', 'end', 'code', 'code_text', 'text'])
    return df


def doc2sent_and_adjust_indices_spacy(row):
    doc = nlp(row['text'])
    for sent in doc.sents:
        sentence_start_index = sent.start_char
        sentence_end_index = sent.end_char

        if row['start'] >= sentence_start_index and row['end'] <= sentence_end_index:
            adjusted_start = row['start'] - sentence_start_index
            adjusted_end = row['end'] - sentence_start_index

            # Extract the sentence text directly from spaCy's Span object
            sentence = sent.text

            # Additional logic to handle discrepancies, similar to before
            extracted_text = sentence.rstrip()[adjusted_start:adjusted_end]
            original_text = row['text'][row['start']:row['end']]

            if extracted_text != original_text:
                print(" original_text: ", original_text)
                print("extracted_text: ", extracted_text)

            return sentence.rstrip(), adjusted_start, adjusted_end

    return row['text'], row['start'], row['end']


def compute_vocabulary_embeddings(isolated_model: torch.nn.Module,
                                  isolated_tokenizer: AutoTokenizer,
                                  vocabulary: List[str],
                                  batch_size: int,
                                  device: str) -> Dict[str, torch.Tensor]:
    """
    Compute embeddings for a given vocabulary using the feature_extractor of the CONORMEN.

    Args:
        isolated_model (torch.nn.Module): Instance of isolated_model.
        isolated_tokenizer (AutoTokenizer): Tokenizer to tokenize the input vocabulary.
        vocabulary (List[str]): List of vocabulary items for which embeddings are to be computed.
        batch_size (int): Size of the batch to be used during computation.
        device (str): Device to which the model needs to be moved for computation ('cuda' or 'cpu').

    Returns:
        Dict[str, torch.Tensor]: A dictionary where keys are vocabulary items and values are their corresponding embeddings.
    """

    MAX_LEN = 512 if isolated_tokenizer.model_max_length > 99999 else None

    def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on the model output.
        """
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    vocabulary_embeddings = {}
    original_device = next(isolated_model.parameters()).device
    device = torch.device(device)
    isolated_model.to(device)

    with torch.no_grad():
        isolated_model.eval()
        num_batches = (len(vocabulary) + batch_size - 1) // batch_size

        for idx in tqdm(range(num_batches), desc="Computing vocabulary embeddings", leave=True):
            start_idx = idx * batch_size
            end_idx = start_idx + batch_size
            batch_vocab = vocabulary[start_idx:end_idx]
            encoded_vocab = isolated_tokenizer(
                batch_vocab, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN).to(device)
            vocab_output = isolated_model(**encoded_vocab)
            vocab_embedding = mean_pooling(
                vocab_output[0], encoded_vocab['attention_mask'])
            for i, vocab_item in enumerate(batch_vocab):
                vocabulary_embeddings[vocab_item] = vocab_embedding[i].cpu().squeeze(
                    0)

    isolated_model.to(original_device)
    torch.cuda.empty_cache()

    return vocabulary_embeddings


class DynamicContextRefining(nn.Module):
    def __init__(self, embedding_dim: int):
        """
        Initialize the Dynamic Context Refining (DCR) module.

        Args:
            embedding_dim (int): The size of the embedding dimension.
        """
        super(DynamicContextRefining, self).__init__()
        self.query_key = nn.Linear(
            embedding_dim, 2 * embedding_dim, bias=False)
        self.embedding_dim = embedding_dim

    def forward(self, isolated_embeddings: torch.Tensor, context_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DCR module.

        Args:
            isolated_embeddings (torch.Tensor): Embeddings from the target entity.
            context_embeddings (torch.Tensor): Embeddings from the broader context surrounding the identified entity.

        Returns:
            torch.Tensor: Updated isolated embeddings after DCR.
        """
        merged_embeddings = torch.stack(
            (isolated_embeddings, context_embeddings), dim=1)
        combined_QK = self.query_key(merged_embeddings)
        Q, K = torch.split(combined_QK, self.embedding_dim, dim=-1)
        W = F.softmax(torch.bmm(Q, K.transpose(1, 2)) /
                      (self.embedding_dim ** 0.5), dim=-1)
        updated_isolated_embeddings = torch.bmm(W, merged_embeddings)[:, 0, :]
        return updated_isolated_embeddings


class CONORMEN(nn.Module):
    def __init__(self, isolated_model, context_model) -> None:
        """
        Initialize the CONORM-EN module.

        Args:
            isolated_model: Pretrained model for target entities.
            context_model: Pretrained model for the broader context.
        """
        super(CONORMEN, self).__init__()

        self.isolated_model = isolated_model

        self.context_model = context_model

        self.DCR = DynamicContextRefining(
            self.isolated_model.config.hidden_size)

        for param in self.isolated_model.parameters():
            param.requires_grad_(False)

        self.isolated_model.eval()

        if self.isolated_model.config.hidden_size != self.context_model.config.hidden_size:
            print(
                "Casting the output of context_model to match the hidden dimension of isolated_model.")
            self.scale = nn.Linear(
                self.context_model.config.hidden_size, self.isolated_model.config.hidden_size)
        else:
            self.scale = None

    def train(self, mode=True):
        """
        Overrides the default train method to always set feature_extractor in eval mode.
        """
        super().train(mode)
        self.isolated_model.eval()

    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on the model output.

        Args:
            token_embeddings (torch.Tensor): Token embeddings from the model.
            attention_mask (torch.Tensor): Mask indicating non-padded elements.

        Returns:
            torch.Tensor: Pooled tensor.
        """
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, encoded_contexts: Dict[str, torch.Tensor], encoded_isolated: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the CONORM-EN module.

        Args:
            encoded_contexts (Dict[str, torch.Tensor]): Encoded input contexts.
            encoded_isolated (Dict[str, torch.Tensor]): Encoded isolated entities.

        Returns:
            torch.Tensor: Updated entity embeddings.
        """
        with torch.no_grad():
            isolated_output = self.isolated_model(**encoded_isolated)
            isolated_embeddings = self.mean_pooling(
                isolated_output[0], encoded_isolated['attention_mask'])

        context_output = self.context_model(**encoded_contexts)
        context_embeddings = self.mean_pooling(
            context_output[0], encoded_contexts['attention_mask'])

        if self.scale:
            context_embeddings = self.scale(context_embeddings)

        updated_embeddings = self.DCR(
            isolated_embeddings, context_embeddings)

        return updated_embeddings


class ConceptNormalizationDataset(Dataset):
    """
    Custom dataset for concept normalization tasks.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing rows of data.
        llt_to_pt_map (Dict[str, str]): Mapping from LLT (Lowest Level Term) to PT (Preferred Term).
        llt_to_llt_text_map (Dict[str, Dict[str, str]]): Mapping from LLT to its textual information.
        vocabulary_embeddings (Dict[str, torch.Tensor]): Precomputed embeddings for vocabulary.
        negative_sampling_ratio (float, optional): Probability to sample a negative instance. Defaults to 0.5.

    Attributes:
        dataframe (pd.DataFrame): The dataframe containing the raw data.
        rows (List[List[Union[str, int]]]): Rows of the dataframe in list format.
        negative_sampling_ratio (float): Probability for negative sampling.
        llt_to_pt_map (Dict[str, str]): LLT to PT mapping.
        llt_to_llt_text_map (Dict[str, Dict[str, str]]): LLT to its textual information.
        all_pt_concepts (List[str]): List of all PT concepts.
        all_llt_texts (List[str]): List of all LLT textual information.
        pt_to_llt_map (Dict[str, List[str]]): Reverse mapping from PT to its LLTs.
        vocabulary_embeddings (Dict[str, torch.Tensor]): Vocabulary embeddings.
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 llt_to_pt_map: Dict[str, str],
                 llt_to_llt_text_map: Dict[str, Dict[str, str]],
                 vocabulary_embeddings: Dict[str, torch.Tensor],
                 negative_sampling_ratio: float = 0.5) -> None:

        # Store the dataframe and convert its rows to a list.
        self.dataframe = dataframe
        self.rows = dataframe.values.tolist()

        # Store other provided configurations.
        self.negative_sampling_ratio = negative_sampling_ratio
        self.llt_to_pt_map = llt_to_pt_map
        self.llt_to_llt_text_map = llt_to_llt_text_map

        # Extract unique PT concepts and LLT text information.
        self.all_pt_concepts = list(set(self.llt_to_pt_map.values()))
        self.all_llt_texts = list(set(llt_to_llt_text_map.values()))

        # Build a reverse mapping from PT to its LLTs.
        self.pt_to_llt_map = {pt: [] for pt in self.all_pt_concepts}
        for llt, pt in self.llt_to_pt_map.items():
            self.pt_to_llt_map[pt].append(llt)

        self.vocabulary_embeddings = vocabulary_embeddings

    def __len__(self) -> int:
        """Return the total number of rows in the dataset."""
        return len(self.dataframe)

    def get_negative_sample(self, positive_llt: str) -> str:
        """
        Fetch a negative sample for a given positive LLT.

        Args:
            positive_llt (str): LLT for which a negative sample is needed.

        Returns:
            str: The text of the negatively sampled concept.
        """
        positive_pt = self.llt_to_pt_map[positive_llt]

        # Randomly select a negative PT different from the positive one.
        negative_pt = random.choice(self.all_pt_concepts)
        while negative_pt == positive_pt:
            negative_pt = random.choice(self.all_pt_concepts)

        # Randomly select an LLT from the list of LLTs associated with the negative PT.
        negative_llt = random.choice(self.pt_to_llt_map[negative_pt])

        return self.llt_to_llt_text_map[negative_llt]

    def __getitem__(self, idx: int) -> Tuple[str, int, int, torch.Tensor, int]:
        """
        Fetch a data item by its index.

        Args:
            idx (int): Index of the data item.

        Returns:
            Tuple[str, int, int, torch.Tensor, int]: Text, start index, end index, vocabulary embedding, and label.
        """
        start, end, positive_llt, concept, text = self.rows[idx]
        start = int(start)
        end = int(end)

        # Decide if the instance should be positive or if a negative sample should be generated.
        label = 1  # default is positive
        if random.random() < self.negative_sampling_ratio:
            concept = self.get_negative_sample(positive_llt)
            label = -1

        vocabulary_embedding = self.vocabulary_embeddings[concept]

        return text, start, end, vocabulary_embedding, label


class CustomLabelEncoder:
    def __init__(self):
        self.label_mapping = {}

    def fit(self, data_list):
        unique_labels = sorted(list(set(data_list)))
        self.label_mapping = {label: idx for idx,
                              label in enumerate(unique_labels)}

    def transform(self, data_list):
        if not self.label_mapping:
            raise ValueError("The encoder has not been fitted yet!")
        return [self.label_mapping.get(label, -1) for label in data_list]

    def inverse_transform(self, encoded_list):
        reverse_mapping = {idx: label for label,
                           idx in self.label_mapping.items()}
        return [reverse_mapping[label] for label in encoded_list]


def annotate_concepts(input_text: List[str], start_indices: List[int], end_indices: List[int]) -> List[str]:
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


def create_encoded_dataframe(val_df, context_tokenizer, isolated_tokenizer, llt_to_llt_text_map, batch_size):
    encoder = CustomLabelEncoder()
    encoder.fit(list(set(llt_to_llt_text_map.values())))

    TEXT_MAX_LEN = 512 if context_tokenizer.model_max_length > 99999 else None
    CONCEPT_MAX_LEN = 512 if isolated_tokenizer.model_max_length > 99999 else None

    # Initialize batches
    batch_starts, batch_ends, batch_texts, batch_llt_texts = [], [], [], []
    batches = []

    for instance in val_df[["start", "end", "text", 'code_text']].values.tolist():
        start, end, text, code_text = instance

        # Collect instances for a batch
        batch_starts.append(start)
        batch_ends.append(end)
        batch_texts.append(text)
        batch_llt_texts.append(code_text)

        # Check if batch is full or if it's the last instance
        if len(batch_starts) == batch_size or instance == val_df[["start", "end", "text", 'code_text']].values.tolist()[-1]:

            # Process Context
            concepts = [t[s:e] for t, s, e in zip(
                batch_texts, batch_starts, batch_ends)]
            encoded_concepts = isolated_tokenizer(
                concepts, padding=True, truncation=True, return_tensors='pt', max_length=TEXT_MAX_LEN)

            # Process Text
            annotated_texts = annotate_concepts(
                batch_texts, batch_starts, batch_ends)
            encoded_texts = context_tokenizer(
                annotated_texts, padding=True, truncation=True, return_tensors='pt', max_length=CONCEPT_MAX_LEN)

            # Process Labels
            encoded_llt_texts = encoder.transform(batch_llt_texts)

            # Append to batches
            batches.append(
                (encoded_concepts, encoded_texts, encoded_llt_texts))

            # Reset for next batch
            batch_starts, batch_ends, batch_texts, batch_llt_texts = [], [], [], []

    return batches, encoder


def process_batch(batch, isolated_tokenizer, context_tokenizer, device):
    """Prepare data and move to the given device."""
    TEXT_MAX_LEN = 512 if context_tokenizer.model_max_length > 99999 else None
    CONCEPT_MAX_LEN = 512 if isolated_tokenizer.model_max_length > 99999 else None

    texts, starts, ends, vocabulary_embeddings, labels = batch
    annotated_texts = annotate_concepts(texts, starts, ends)

    encoded_texts = context_tokenizer(
        annotated_texts, padding=True, truncation=True, return_tensors='pt', max_length=CONCEPT_MAX_LEN)
    concepts = [text[start:end]
                for text, start, end in zip(texts, starts, ends)]
    encoded_concepts = isolated_tokenizer(
        concepts, padding=True, truncation=True, return_tensors='pt', max_length=TEXT_MAX_LEN)

    # Use a dictionary comprehension and context to move to device
    encoded_texts = {k: v.to(device) for k, v in encoded_texts.items()}
    encoded_concepts = {k: v.to(device) for k, v in encoded_concepts.items()}

    vocabulary_embeddings = vocabulary_embeddings.to(device)
    labels = labels.to(device)

    return encoded_texts, encoded_concepts, vocabulary_embeddings, labels


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip, isolated_tokenizer, context_tokenizer):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, leave=False, desc="Epoch"):
        encoded_texts, encoded_concepts, vocabulary_embeddings, labels = process_batch(
            batch, isolated_tokenizer, context_tokenizer, device)

        optimizer.zero_grad()
        embeddings = model(encoded_texts, encoded_concepts)
        loss = criterion(embeddings, vocabulary_embeddings, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_one_epoch(model, f1_val_df, val_labelencoder, vocabulary_embeddings, device, n_values=[1, 3, 5, 10, 100]):
    """Validate the model on given data and compute Accuracy@n."""
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(f1_val_df, leave=False, desc="Validating"):
            encoded_concepts, encoded_texts, labels = batch
            encoded_texts = {k: v.to(device) for k, v in encoded_texts.items()}
            encoded_concepts = {k: v.to(device)
                                for k, v in encoded_concepts.items()}
            embeddings = model(encoded_texts, encoded_concepts)
            embeddings_list.append(embeddings)
            labels_list.extend(labels)

    embeddings_list = F.normalize(torch.cat(embeddings_list).cpu(), p=2, dim=1)
    labels_list = np.array(labels_list)

    embedding_matrix = torch.stack(
        [tensor for tensor in vocabulary_embeddings.values()], dim=0)
    embedding_matrix = F.normalize(embedding_matrix, p=2, dim=1)

    result = torch.mm(embeddings_list, embedding_matrix.t())

    accuracy_at_n = {}
    keys = list(vocabulary_embeddings.keys())  # List of keys for lookup

    for n in n_values:
        correct_predictions = 0
        for true_label, similarities in zip(labels_list, result):
            top_n_indices = torch.topk(similarities, n, largest=True).indices
            highest_inner_product_keys = [keys[index]
                                          for index in top_n_indices]
            predicted_labels = val_labelencoder.transform(
                highest_inner_product_keys)

            if true_label in predicted_labels:
                correct_predictions += 1

        accuracy_at_n[f"Acc@{n}"] = correct_predictions / len(labels_list)

    return accuracy_at_n