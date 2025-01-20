from src.model_helpers import *
import argparse
import shutil
import os

print()

def main() -> None:
    """
    Train a NER model given a config file.

    Returns:
        None
    """

    print()
    print("Training model...")
    print()
    
    # Load the config and data
    parser = argparse.ArgumentParser(description='Train a NER model given a config file')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    
    args = parser.parse_args()
    
    # Read configuration settings
    with open(args.config_path, 'r') as config_file:
        config = json.load(config_file)

    data_path = os.path.dirname(config['train_standoff_path'])
    device = config["device"]
    pretrained_model = config["pretrained_model"]
    max_len = config["max_len"]
    two_phase_learning = config["two_phase_learning"]
    max_epoch = config["max_epoch"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    warmup_proportion = config["warmup_proportion"]
    post_process_type = config["post_process_type"]
    split_brat_on_newlines = config["split_brat_on_newlines"]
    
    # Determine the folder path based on the value of BIO
    bio_path = os.path.join(data_path, "bio")
    
    # Define the data groups to use
    data_groups = ['train', 'val', 'test']
    
    # Load the text, labels, and tag2idx using the defined folder path and data groups
    text, labels, starts, ends, file_names, columns = load_data(bio_path, data_groups)
    
    # Create a dictionary for converting tag to index
    tag2idx = create_tag2idx(columns)
    
    # Create path for saving the logs
    PATH = create_path(
        folder_path=bio_path,
        pretrained_model=pretrained_model,
    )
    
    # Create the temporary folder for storing validation predictions in the required format
    for_loop_folder = os.path.join(PATH, "temp", "for_loop_preds")
    if not os.path.exists(for_loop_folder):
        os.makedirs(for_loop_folder)
    
    # Create the temporary folder for storing test predictions in the required format
    test_prediction_folder = os.path.join(PATH, "temp", "test_prediction")
    if not os.path.exists(test_prediction_folder):
        os.makedirs(test_prediction_folder)
    
    # Create a SummaryWriter object and save the config, tag2idx and columns
    writer = SummaryWriter(PATH)
    writer.add_text('config', json.dumps(config, indent=4), 0)
    save_config(config, os.path.join(PATH, "used_config.json"))
    save_config(tag2idx, os.path.join(PATH, "tag2idx.json"))
    save_list(columns, os.path.join(PATH, "columns.pickle"))
    shutil.copyfile(os.path.join(data_path, 'tokenizer.pickle'), os.path.join(PATH, 'tokenizer.pickle'))
    
    # Initialize the tokenizer with the pretrained model and set do_lower_case to False
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    # Align the data for BERT-like training by splitting the text and labels into subwords,
    # aligned_text and aligned_labels are the equivalent for BERT-like training
    print()
    aligned_text, aligned_labels, aligned_subword_dummies, aligned_starts, aligned_ends, aligned_file_names = align_data(data_groups,
                                                                                                                         text,
                                                                                                                         labels,
                                                                                                                         starts,
                                                                                                                         ends,
                                                                                                                         file_names,
                                                                                                                         tokenizer,
                                                                                                                         max_len)
    
    # Create a data loader for the training data
    print()
    train_dataloader, train_str2num_file_names = create_dataloader("train",
                                                                   aligned_text,
                                                                   aligned_labels,
                                                                   aligned_subword_dummies,
                                                                   aligned_starts,
                                                                   aligned_ends,
                                                                   aligned_file_names,
                                                                   tokenizer,
                                                                   tag2idx,
                                                                   max_len,
                                                                   batch_size)
    
    # Create a data loader for the validation data
    dev_dataloader, dev_str2num_file_names = create_dataloader("val",
                                                               aligned_text,
                                                               aligned_labels,
                                                               aligned_subword_dummies,
                                                               aligned_starts,
                                                               aligned_ends,
                                                               aligned_file_names,
                                                               tokenizer,
                                                               tag2idx,
                                                               max_len,
                                                               batch_size)
    
    # Create a data loader for the test data
    test_dataloader, test_str2num_file_names = create_dataloader("test",
                                                                 aligned_text,
                                                                 aligned_labels,
                                                                 aligned_subword_dummies,
                                                                 aligned_starts,
                                                                 aligned_ends,
                                                                 aligned_file_names,
                                                                 tokenizer,
                                                                 tag2idx,
                                                                 max_len,
                                                                 batch_size)
    
    
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    
    # Create an instance of the token classification model
    model = MyAutoModelForTokenClassification(pretrained_model_name_or_path=pretrained_model,
                                              tag2idx=tag2idx,
                                              columns=columns)
    
    
    # Move the model to the specified device
    model.to(device)
    
    if two_phase_learning:
        # Create a separate optimizer and scheduler for the first epoch
        optimizer = AdamW(model.parameters(), lr=0.001, eps=1e-8)
        total_steps = len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_proportion),
            num_training_steps=total_steps
        )
    else:
        # Create an AdamW optimizer
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    
        # Calculate the total number of steps for training
        total_steps = len(train_dataloader) * max_epoch
    
        # Create a scheduler for the optimizer
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_proportion),
            num_training_steps=total_steps
        )
    
    # Initialize best_f1 to 0
    best_f1 = 0
    test_f1 = 0
    f1_history = []
    
    # Start the timer
    start_time = time.time()
    
    # Loop through the number of maximum epochs
    pbar = trange(max_epoch, desc="Epoch", leave=False)
    for epoch in pbar:
    
        if two_phase_learning and epoch == 0:
            # Freeze the language model weights during the first epoch
            model.train()
            model.language_model.eval()
            for param in model.language_model.parameters():
                param.requires_grad = False
    
        elif two_phase_learning and epoch == 1:
            # Unfreeze the language model weights for the remaining epochs
            model.train()
            model.language_model.train()
            for param in model.language_model.parameters():
                param.requires_grad = True
    
            # Create a new optimizer and scheduler for the remaining epochs
            optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
            total_steps = len(train_dataloader) * (max_epoch - 1)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_steps * warmup_proportion),
                num_training_steps=total_steps
            )
        else:
            model.train()
    
        # Perform a training step and get the average train loss
        avg_train_loss = train_step(train_dataloader,
                                    model,
                                    optimizer,
                                    scheduler,
                                    device)
    
        # Perform an evaluation step and get the reports
        model.eval()
        report = evaluation_step(dev_dataloader, model, device, columns, tag2idx,
                                 dev_str2num_file_names,
                                 os.path.join(PATH, "temp", "for_loop_preds"),
                                 os.path.join(data_path, "val"),
                                 os.path.join(data_path, "val"),
                                 post_process_type=post_process_type,
                                 split_brat_on_newlines=split_brat_on_newlines)
        
    
        # Add the metrics for each entity to the tensorboard writer
        for entity, metrics in report.items():
            for metric, value in metrics.items():
                writer.add_scalar(f'{entity}/{metric}', value, epoch+1)
    
        # Compare the f1-score of the current epoch to the best f1-score
        # Save the model if it has a better f1-score
        f1 = report["Overall"]["F1"]
        f1_history.append(f1)
        report["meta"] = {"avg_increase": average_increase(
            f1_history), "epoch": epoch + 1}
    
        if f1 > best_f1:
            best_f1 = f1
            save_to_json(report, os.path.join(PATH, "best_model_validation_report.json"))
            torch.save(model.state_dict(), os.path.join(PATH, "best_model.pt"))
    
        # Update the tqdm progress bar with the F1-score for the current epoch
        pbar.set_postfix({"Val F1-score": best_f1})
    
    # Calculate the elapsed time
    training_time_s = time.time() - start_time
    epoch_time_s = training_time_s/max_epoch
    training_time = timedelta(seconds=int(training_time_s))
    epoch_time = timedelta(seconds=int(epoch_time_s))
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(PATH, "best_model.pt")))
    model.to(device)
    model.eval()
    
    # Evaluate the model on the test set
    test_report = evaluation_step(test_dataloader, model, device, columns, tag2idx,
                                  test_str2num_file_names,
                                  os.path.join(PATH, "temp", "test_prediction"),
                                  os.path.join(data_path, "test"),
                                  os.path.join(data_path, "test"),
                                  post_process_type=post_process_type,
                                  split_brat_on_newlines=split_brat_on_newlines)
    
    # Add the metrics for each entity to the tensorboard writer for the test set
    for entity, metrics in test_report.items():
        for metric, value in metrics.items():
            writer.add_scalar(f'{entity}/test_{metric}', value, epoch+1)
    
    # Save the test report to a JSON file
    save_to_json(test_report, os.path.join(PATH, "test_report.json"))
    
    # Close the tensorboard writer
    writer.close()
    
    # Summary
    print("     Val  F1-score:", best_f1)
    print("     Test F1-score:", test_report["Overall"]["F1"], "\n")
    print("     Training time:", training_time)
    print("Average epoch time:", epoch_time, "\n")

    shutil.rmtree(os.path.join(PATH, "temp"))

if __name__ == "__main__":
    main()