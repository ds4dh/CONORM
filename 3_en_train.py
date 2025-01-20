import argparse
import os
from src.conorm_en_helpers import *

def main(config_path):
    config = load_config(config_path)
    
    # Accessing each variable
    llt_asc_path = config['llt_asc_path']
    pt_asc_path = config['pt_asc_path']
    train_standoff_path = config['train_standoff_path']
    val_standoff_path = config['val_standoff_path']
    test_standoff_path = config['test_standoff_path']
    meddra_level = config['meddra_level']
    isolated_model_name = config['isolated_model_name']
    context_model_name = config['context_model_name']
    document_level = config['document_level']
    epochs = config['epochs']
    device = config['device']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    grad_clip = config['grad_clip']
    margin = config['margin']
    
    PATH = create_path(train_standoff_path,
                       isolated_model_name, context_model_name)
    
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    save_config(config, os.path.join(PATH, "used_config.json"))
    
    llt_dict, llt_to_pt, pt_dict, pt_to_pt = build_en_dict_from_MedDRA(
        llt_asc_path,
        pt_asc_path
    )
    
    if meddra_level == "llt":
        code_dict = llt_dict.copy()
        code_to_higher = llt_to_pt.copy()
        del llt_dict, llt_to_pt, pt_dict, pt_to_pt
    elif meddra_level == "pt":
        code_dict = pt_dict.copy()
        code_to_higher = pt_to_pt.copy()
        del llt_dict, llt_to_pt, pt_dict, pt_to_pt
    else:
        print(f'meddra_level should be either "llt" or "pt".')
    
    train_df = create_dataframe_from_ann_files(
        train_standoff_path,
        parse_type=meddra_level,
        code_dict=code_dict
    )
    
    if document_level:
        pass
    else:
        train_df[['text', 'start', 'end']] = train_df.parallel_apply(
            lambda row: pd.Series(doc2sent_and_adjust_indices_spacy(row)), axis=1)
    
    val_df = create_dataframe_from_ann_files(
        val_standoff_path,
        parse_type=meddra_level,
        code_dict=code_dict
    )
    if document_level:
        pass
    else:
        val_df[['text', 'start', 'end']] = val_df.parallel_apply(
            lambda row: pd.Series(doc2sent_and_adjust_indices_spacy(row)), axis=1)
    
    test_df = create_dataframe_from_ann_files(
        test_standoff_path,
        parse_type=meddra_level,
        code_dict=code_dict
    )
    if document_level:
        pass
    else:
        test_df[['text', 'start', 'end']] = test_df.parallel_apply(
            lambda row: pd.Series(doc2sent_and_adjust_indices_spacy(row)), axis=1)
    
    isolated_tokenizer = AutoTokenizer.from_pretrained(isolated_model_name)
    isolated_model = AutoModel.from_pretrained(isolated_model_name)
    
    all_code_texts = list(code_dict.values())
    
    vocabulary_embeddings = compute_vocabulary_embeddings(isolated_model,
                                                          isolated_tokenizer,
                                                          all_code_texts,
                                                          batch_size,
                                                          device)
    
    context_tokenizer = AutoTokenizer.from_pretrained(context_model_name)
    context_tokenizer.add_tokens(["<START_ENTITY>", "<END_ENTITY>"])
    context_model = AutoModel.from_pretrained(context_model_name)
    context_model.resize_token_embeddings(len(context_tokenizer))
    
    # Model and training objects
    model = CONORMEN(isolated_model=isolated_model, context_model=context_model)
    criterion = nn.CosineEmbeddingLoss(margin=margin)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    train_dataset = ConceptNormalizationDataset(
        train_df, code_to_higher, code_dict, vocabulary_embeddings)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    f1_val_df, val_labelencoder = create_encoded_dataframe(
        val_df, context_tokenizer, isolated_tokenizer, code_dict, batch_size=batch_size)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=learning_rate,
                                              epochs=epochs,
                                              steps_per_epoch=len(train_dataset)//batch_size)
    
    model.to(device)
    criterion = criterion.to(device)
    progress_bar = trange(epochs, desc="Training", leave=True)
    
    best_acc1 = 0.0  # Initialize the best F1-score
    best_model_state = None  # Initialize the best model state
    
    for epoch in progress_bar:
        train_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device, grad_clip, isolated_tokenizer, context_tokenizer)
        results = validate_one_epoch(
            model, f1_val_df, val_labelencoder, vocabulary_embeddings, device)
    
        acc1 = results['Acc@1']
    
        # If the current F1-score is better than the best seen so far, store the model's state
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_model_state = model.state_dict().copy()  # store a copy of the model state
            cpu_model_state = {key: value.cpu()
                               for key, value in best_model_state.items()}
            torch.save(cpu_model_state, os.path.join(PATH, "best_model.pt"))
            save_config(results, os.path.join(
                PATH, "best_model_validation_report.json"))
    
        # Update the progress bar description with the current epoch's losses.
        progress_bar.set_description(
            f"{epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc@1: {acc1:.4f} | Best Acc@1: {best_acc1:.4f} |")
        scheduler.step()
    
    model.load_state_dict(best_model_state)
    model.eval()
    f1_test_df, _ = create_encoded_dataframe(
        test_df, context_tokenizer, isolated_tokenizer, code_dict, batch_size=batch_size)
    test_results = validate_one_epoch(
        model, f1_test_df, val_labelencoder, vocabulary_embeddings, device)
    save_config(test_results, os.path.join(PATH, "test_report.json"))
    print(f"Test Acc@1 = {test_results['Acc@1']}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a EN model given a config file')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    
    args = parser.parse_args()
    
    main(args.config_path)