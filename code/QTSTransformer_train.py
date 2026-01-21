import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import random
import time
import math
from tqdm import tqdm
from typing import Any, Optional, Tuple, Callable
from collections import Counter
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, recall_score, confusion_matrix

from QTSTransformer import QuantumTSTransformer


def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_weights(model: torch.nn.Module) -> None:
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    model.apply(_init_weights)
    

def load_fmri_data(dataset, parcel_type, phenotypes_to_include, target_phenotype, categorical_features):
    """
    Load and combine fMRI data and phenotype data for all subjects.

    Args:
        dataset (str): Type of dataset to use (ABCD, UKB).
        parcel_type (str): Parcellation type for each subject's fMRI data (HCP, HCP180, Schaefer).
        phenotypes_to_include (list): Phenotype columns to include as input features.
        target_phenotype (str): Phenotype column to use as the target.
        categorical_features (list): Columns in phenotypes_to_include that are categorical.

    Returns:
        X (list of tensors): List of input tensors (fMRI + phenotypes) for all subjects.
        y (list): List of target labels for all subjects.
    """
    # Identifiers for sites (for stratification)
    sites = None 

    # Load phenotype data
    if dataset=="ABCD":
        phenotypes = pd.read_csv("ABCD/ABCD_phenotype_total.csv")
        # Drop Subjects with Missing Values
        phenotypes = phenotypes[phenotypes_to_include+["subjectkey", target_phenotype]].dropna()
        subject_ids = phenotypes["subjectkey"].values
    elif dataset=="UKB":
        phenotypes = pd.read_csv("UKB/UKB_phenotype_gps_fluidint.csv")
        # Drop Subjects with Missing Values
        phenotypes = phenotypes[phenotypes_to_include+["eid", target_phenotype]].dropna()
        subject_ids = phenotypes["eid"].values   
    elif dataset == "ENIGMA_OCD":
        phenotypes = pd.read_csv("/pscratch/sd/p/pakmasha/enigma_quantum/metadata/ENIGMA_QC_final_subject_list.csv")
        
        # Ensure Site is included if available
        cols_to_use = phenotypes_to_include + ["Unique_ID", target_phenotype]
        if "Sample" not in cols_to_use:
             cols_to_use.append("Sample")
             
        phenotypes = phenotypes[cols_to_use].dropna()
        subject_ids = phenotypes["Unique_ID"].values
        if "Sample" in phenotypes.columns:
            sites = phenotypes["Sample"].values
    
    # Identify continuous features to normalize
    # continuous_features = [col for col in phenotypes_to_include if col not in categorical_features]
        
    # Select input phenotypes and target
    input_phenotypes = phenotypes[phenotypes_to_include].values
    target_labels = phenotypes[target_phenotype].values      

    X, y, valid_subject_ids, valid_sites = [], [], [], []
    valid_subject_count = 0
    
    # Load fMRI data for each subject
    for i, subject_id in enumerate(subject_ids):
        if dataset=="ABCD":
            if parcel_type=="HCP":
                fmri_path = f"ABCD/sub-{subject_id}/hcp_mmp1_sub-{subject_id}.npy"
            elif parcel_type=="HCP180":
                fmri_path = f"ABCD/sub-{subject_id}/hcp_mmp1_180_sub-{subject_id}.npy"
            elif parcel_type=="Schaefer":
                fmri_path = f"ABCD/sub-{subject_id}/schaefer_sub-{subject_id}.npy"                
        elif dataset=="UKB":
            if parcel_type=="HCP":
                fmri_path = f"UKB/{subject_id}/hcp_mmp1_{subject_id}.npy"
            elif parcel_type=="HCP180":
                fmri_path = f"UKB/{subject_id}/hcp_mmp1_{subject_id}.npy"
            elif parcel_type=="Schaefer":
                fmri_path = f"UKB/{subject_id}/schaefer_400Parcels_17Networks_{subject_id}.npy"
        elif dataset=="ENIGMA_OCD":
            fmri_path = f"/pscratch/sd/p/pakmasha/enigma_quantum_data/{subject_id}/{subject_id}.npy"
            
        if not os.path.exists(fmri_path):
            # print(f"Missing fMRI file for subject {subject_id}. Skipping...")
            continue

        fmri_data = np.load(fmri_path)  # Shape: (time_points, brain_regions)

        # Truncate UKB data to 363 time points
        if dataset == "UKB" and fmri_data.shape[0] > 363:
            start_idx = (fmri_data.shape[0] - 363) // 2  # Calculate starting index for truncation
            fmri_data = fmri_data[start_idx:start_idx + 363]
        
        if dataset == "ENIGMA_OCD":
            if fmri_data.shape[0] != 115:
                raise ValueError(f"Subject {subject_id} has sequence length {fmri_data.shape[0]}, expected 115.")
                
        fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32)

        # Stack phenotype features as additional columns across all time points
        phenotype_tensor = torch.tensor(input_phenotypes[i], dtype=torch.float32).repeat(fmri_tensor.shape[0], 1)
        combined_features = torch.cat((fmri_tensor, phenotype_tensor), dim=1)  # Shape: (time_points, brain_regions + phenotypes)

        X.append(combined_features)
        y.append(target_labels[i])  # Target is one label per subject
        valid_subject_ids.append(subject_id)
        if sites is not None:
             valid_sites.append(sites[i])
        valid_subject_count += 1

        
    print(f"Final sample size (number of subjects): {valid_subject_count}")
    
    return X, y, valid_subject_ids, valid_sites


def split_and_prepare_dataloaders(X, y, subject_ids, batch_size, sequence_length, device, binary, split_output_path, stratify=True, sites=None):
    """
    Split data into train, validation, and test sets and create DataLoaders with sliding windows.

    Args:
        X (list of tensors): Input data for all subjects (time-series data combined with phenotypes).
        y (list): Target labels for all subjects.
        batch_size (int): Batch size for DataLoader.
        sequence_length (int): Length of each input sequence.
        device (torch.device): Device to use (CPU or GPU).
        stratify (bool): Whether to stratify the split by target labels.
        sites (list, optional): List of site IDs for each subject to allow site-stratification.

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for train, validation, and test sets.
    """
    def create_sequences(data, labels):
        """
        Create fixed-length sequences for each subject using a sliding window.

        Args:
            data (list of tensors): Time-series data for all subjects.
            labels (list): Target labels for all subjects.

        Returns:
            sequences (list of tensors): Sequences of shape (sequence_length, feature_dim).
            sequence_labels (list): Labels for each sequence.
        """
        sequences, sequence_labels = [], []

        for subject_data, label in zip(data, labels):
            num_time_points = subject_data.shape[0]
            for start in range(0, num_time_points - sequence_length + 1, sequence_length):
                seq = subject_data[start:start + sequence_length]  # Fixed-length sequence
                sequences.append(seq)
                sequence_labels.append(label)  # Use the same label for all sequences from this subject

        return sequences, sequence_labels

    # Convert lists to numpy arrays for splitting
    y = np.array(y)
    subject_ids = np.array(subject_ids)
    # Prepare stratification labels
    stratify_labels = None
    if stratify:
        if sites is not None and len(sites) == len(y):
            # Combined stratification (Target + Site)
            stratify_labels = [f"{lbl}_{site}" for lbl, site in zip(y, sites)]
            
            # Check for rare combinations (need at least 2 for splitting)
            counts = Counter(stratify_labels)
            if min(counts.values()) < 2:
                print("Warning: Some site-target combinations have fewer than 2 samples. Falling back to target-only stratification.")
                stratify_labels = y
        else:
            stratify_labels = y

    # Split into train, validation, and test sets
    # We include sites in the split input args to preserve them for the second split if needed
    split_args = [X, y, subject_ids]
    if sites is not None and len(sites) == len(y):
         split_args.append(sites)
         
    split_results_1 = train_test_split(
        *split_args, test_size=0.3, stratify=stratify_labels, random_state=42
    )
    
    train_X = split_results_1[0]
    temp_X = split_results_1[1]
    train_y = split_results_1[2]
    temp_y = split_results_1[3]
    train_subs = split_results_1[4]
    temp_subs = split_results_1[5]
    
    temp_sites = None
    if sites is not None and len(sites) == len(y):
        # train_sites = split_results_1[6]
        temp_sites = split_results_1[7]
    
    # Second split (Validation vs Test)
    stratify_labels_2 = None
    if stratify:
         if temp_sites is not None:
              stratify_labels_2 = [f"{lbl}_{site}" for lbl, site in zip(temp_y, temp_sites)]
              counts = Counter(stratify_labels_2)
              if min(counts.values()) < 2:
                  print("Warning (2nd split): Some site-target combinations have fewer than 2 samples. Falling back to target-only stratification.")
                  stratify_labels_2 = temp_y
         else:
              stratify_labels_2 = temp_y
              
    split_args_2 = [temp_X, temp_y, temp_subs]
    if temp_sites is not None:
         split_args_2.append(temp_sites)

    split_results_2 = train_test_split(
        *split_args_2, test_size=0.5, stratify=stratify_labels_2, random_state=42
    )
    
    val_X = split_results_2[0]
    test_X = split_results_2[1]
    val_y = split_results_2[2]
    test_y = split_results_2[3]
    val_subs = split_results_2[4]
    test_subs = split_results_2[5]

    # Save splits
    if split_output_path:
        os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
        with open(split_output_path, "w") as f:
            f.write("train_subjects\n")
            for sub in train_subs:
                f.write(f"{sub}\n")
            f.write("val_subjects\n")
            for sub in val_subs:
                f.write(f"{sub}\n")
            f.write("test_subjects\n")
            for sub in test_subs:
                f.write(f"{sub}\n")
        print(f"Splits saved to {split_output_path}")

    # # Subject-wise normalization
    # def normalize_subject_wise(subjects):
    #     normalized = []
    #     for subj in subjects:
    #         mean = subj.mean(dim=0, keepdim=True)
    #         std = subj.std(dim=0, keepdim=True)
    #         std[std == 0] = 1e-8
    #         normalized.append((subj - mean) / std)
    #     return normalized

    # train_X = normalize_subject_wise(train_X)
    # val_X = normalize_subject_wise(val_X)
    # test_X = normalize_subject_wise(test_X)

    # Original split-level normalization
    # Concatenate all subjects' data along the time dimension for normalization
    train_X_concat = torch.cat(train_X, dim=0)  # shape: (total_time_points_all_subjects, num_features)
    # Compute mean/std from training set ONLY
    train_X_mean = train_X_concat.mean(dim=0, keepdim=True)
    train_X_std = train_X_concat.std(dim=0, keepdim=True)
    train_X_std[train_X_std == 0] = 1e-8  # Avoid division by zero
    
    def normalize_subjects(subjects, mean, std):
        return [(subj - mean) / std for subj in subjects]
        
    train_X = normalize_subjects(train_X, train_X_mean, train_X_std)
    val_X = normalize_subjects(val_X, train_X_mean, train_X_std)
    test_X = normalize_subjects(test_X, train_X_mean, train_X_std)
    
    # Create fixed-length sequences for each split
    train_sequences, train_sequence_labels = create_sequences(train_X, train_y)
    val_sequences, val_sequence_labels = create_sequences(val_X, val_y)
    test_sequences, test_sequence_labels = create_sequences(test_X, test_y)

    # Convert to PyTorch tensors and create DataLoaders
    def create_dataloader(sequences, labels):
        x_tensors = [seq.to(device) for seq in sequences]
        y_tensor = torch.tensor(labels, dtype=torch.float32 if binary else torch.long, device=device)        
        dataset = TensorDataset(torch.stack(x_tensors), y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_dataloader(train_sequences, train_sequence_labels)
    val_loader = create_dataloader(val_sequences, val_sequence_labels)
    test_loader = create_dataloader(test_sequences, test_sequence_labels)

    return train_loader, val_loader, test_loader


def create_model(
    hyperparams: dict[str, Any], device: torch.device, sequence_length: int, feature_dim: int
) -> torch.nn.Module:
    model = QuantumTSTransformer(
        n_qubits=hyperparams["qubits"],
        n_timesteps=sequence_length,
        degree=hyperparams["degree"],
        n_ansatz_layers=hyperparams["ansatz_layers"],
        feature_dim=feature_dim,
        output_dim=hyperparams["output_dim"],
        dropout=hyperparams["dropout"],
        device=device,
        rotation_scale=hyperparams.get("rotation_scale", 1.0)
    )
    return model


def train_epoch(
    model: torch.nn.Module,
    iterator: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    clip: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
):
    model.train()
    epoch_loss = 0
    y_true, y_scores = [], []  # For AUC calculation
    
    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat.squeeze(), y)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler:
            scheduler.step()
        epoch_loss += loss.item()
        
        # Convert predictions to probabilities if using BCEWithLogitsLoss
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            yhat = torch.sigmoid(yhat)
            
        # Ensure correct format for AUC calculation
        y_true.extend(y.cpu().numpy().astype(int))  # Convert to int for binary labels
        y_scores.extend(yhat.detach().cpu().squeeze().numpy())
    
    try:
        # Convert to numpy arrays and ensure proper shape
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Check if we have binary classification
        unique_labels = np.unique(y_true)
        if len(unique_labels) != 2:
            print(f"Warning: Found {len(unique_labels)} classes. AUC requires binary classification.")
            auc = None
        else:
            auc = roc_auc_score(y_true, y_scores)
    except ValueError as e:
        print(f"AUC calculation error: {e}")
        print(f"y_true shape: {np.shape(y_true)}, unique values: {np.unique(y_true)}")
        print(f"y_scores shape: {np.shape(y_scores)}, range: [{np.min(y_scores)}, {np.max(y_scores)}]")
        auc = None
        
    # Calculate additional metrics for binary classification
    if unique_labels is not None and len(unique_labels) == 2 and auc is not None:
        preds = (y_scores >= 0.5).astype(int)
        bal_acc = balanced_accuracy_score(y_true, preds)
        sens = recall_score(y_true, preds, pos_label=1, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        bal_acc, sens, spec = None, None, None
        tn, fp, fn, tp = None, None, None, None

    return epoch_loss / len(iterator), auc, bal_acc, sens, spec, tn, fp, fn, tp

def evaluate(
    model: torch.nn.Module,
    iterator: DataLoader,
    criterion,
):
    model.eval()
    epoch_loss = 0
    y_true, y_scores = [], []
    
    with torch.no_grad():
        for x, y in tqdm(iterator):
            yhat = model(x)
            loss = criterion(yhat.squeeze(), y)
            epoch_loss += loss.item()
            
            # Convert predictions to probabilities if using BCEWithLogitsLoss
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                yhat = torch.sigmoid(yhat)
                
            # Ensure correct format for AUC calculation
            y_true.extend(y.cpu().numpy().astype(int))  # Convert to int for binary labels
            y_scores.extend(yhat.cpu().squeeze().numpy())
    
    try:
        # Convert to numpy arrays and ensure proper shape
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Check if we have binary classification
        unique_labels = np.unique(y_true)
        if len(unique_labels) != 2:
            print(f"Warning: Found {len(unique_labels)} classes. AUC requires binary classification.")
            auc = None
        else:
            auc = roc_auc_score(y_true, y_scores)
    except ValueError as e:
        print(f"AUC calculation error: {e}")
        print(f"y_true shape: {np.shape(y_true)}, unique values: {np.unique(y_true)}")
        print(f"y_scores shape: {np.shape(y_scores)}, range: [{np.min(y_scores)}, {np.max(y_scores)}]")
        auc = None
        
    # Calculate additional metrics for binary classification
    if unique_labels is not None and len(unique_labels) == 2 and auc is not None:
        preds = (y_scores >= 0.5).astype(int)
        bal_acc = balanced_accuracy_score(y_true, preds)
        sens = recall_score(y_true, preds, pos_label=1, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        bal_acc, sens, spec = None, None, None
        tn, fp, fn, tp = None, None, None, None
        
    return epoch_loss / len(iterator), auc, bal_acc, sens, spec, tn, fp, fn, tp


def train_epoch_multiclass(
    model: torch.nn.Module,
    iterator: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    clip: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
):
    model.train()
    epoch_loss = 0
    y_true, y_pred_classes = [], []  # For accuracy calculation

    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        yhat = model(x)

        loss = criterion(yhat.squeeze(), y)
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()

        # Collect predictions and true labels for accuracy
        _, predicted_classes = torch.max(yhat, dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred_classes.extend(predicted_classes.cpu().numpy())

    # Calculate accuracy
    # acc = accuracy_score(y_true, y_pred_classes)         
    try:
    # Train the model and compute roc_auc_score
        acc = accuracy_score(y_true, y_pred_classes)
    except ValueError as e:
        print(f"Skipping hyperparameter set due to error: {e}")
        auc = None  # Or any fallback value           
        
    return epoch_loss / len(iterator), acc


def evaluate_multiclass(
    model: torch.nn.Module,
    iterator: DataLoader,
    criterion,
):
    model.eval()
    epoch_loss = 0
    y_true, y_pred_classes = [], []  # For accuracy calculation
    
    with torch.no_grad():
        for x, y in tqdm(iterator):
            yhat = model(x)
            loss = criterion(yhat.squeeze(), y)
            epoch_loss += loss.item()

            # Collect predictions and true labels for accuracy
            _, predicted_classes = torch.max(yhat, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred_classes.extend(predicted_classes.cpu().numpy())

    # acc = accuracy_score(y_true, y_pred_classes)         
    try:
    # Train the model and compute roc_auc_score
        acc = accuracy_score(y_true, y_pred_classes)
    except ValueError as e:
        print(f"Skipping hyperparameter set due to error: {e}")
        auc = None  # Or any fallback value   
        
    return epoch_loss / len(iterator), acc


def train_cycle(
    model: torch.nn.Module,
    hyperparams: dict[str, Any],
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    tuning_set,
):
    if hyperparams["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
        )
    elif hyperparams["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
        )    
    elif hyperparams["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
        )
        
    scheduler = None
    if hyperparams["lr_sched"] == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=hyperparams["restart_epochs"]
        )

    if hyperparams["binary"]==True:
        criterion = torch.nn.BCEWithLogitsLoss()  # Loss function for binary classification
    else:
        criterion = torch.nn.CrossEntropyLoss()   # Loss function for multi-class classification

    # Lists to store metrics
    train_metrics, valid_metrics, test_metrics = [], [], []
    
    best_valid_loss = float("inf")
    best_valid_auc = 0.0 # Initialize best AUC
    best_valid_acc = 0.0 # Initialize best accuracy for multiclass
    patience = hyperparams.get("patience", 10)
    patience_counter = 0
    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()

        if hyperparams["binary"]==True:
            train_loss, train_auc, train_bal_acc, train_sens, train_spec, train_tn, train_fp, train_fn, train_tp = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            hyperparams["max_grad_norm"],
            scheduler,
            )

            valid_loss, valid_auc, valid_bal_acc, valid_sens, valid_spec, valid_tn, valid_fp, valid_fn, valid_tp = evaluate(model, val_loader, criterion)
        
        else:
            train_loss, train_acc = train_epoch_multiclass(
            model,
            train_loader,
            optimizer,
            criterion,
            hyperparams["max_grad_norm"],
            scheduler,
            )

            valid_loss, valid_acc = evaluate_multiclass(model, val_loader, criterion)            

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if hyperparams["binary"]:
                 best_valid_auc = valid_auc
            else:
                 best_valid_acc = valid_acc
            patience_counter = 0
            
            # Define output directory and ensure it exists
            output_dir = "/pscratch/sd/p/pakmasha/enigma_quantum/model"
            os.makedirs(output_dir, exist_ok=True)
            
            if hyperparams["model_dir"]==None:
                torch.save(model.state_dict(), f"{output_dir}/best_time_series_model.pt")
            else:
                torch.save(model.state_dict(), f"{output_dir}/{hyperparams['model_dir']}_{hyperparams['seed']}_{tuning_set}.pt")
        else:
            patience_counter += 1

        # Append train and validation metrics for each epoch
        if hyperparams["binary"]:
            train_metrics.append({
                'epoch': epoch + 1, 'train_loss': train_loss, 'train_auc': train_auc,
                'train_bal_acc': train_bal_acc, 'train_sens': train_sens, 'train_spec': train_spec,
                'train_tn': train_tn, 'train_fp': train_fp, 'train_fn': train_fn, 'train_tp': train_tp
            })
            valid_metrics.append({
                'epoch': epoch + 1, 'valid_loss': valid_loss, 'valid_auc': valid_auc,
                'valid_bal_acc': valid_bal_acc, 'valid_sens': valid_sens, 'valid_spec': valid_spec,
                'valid_tn': valid_tn, 'valid_fp': valid_fp, 'valid_fn': valid_fn, 'valid_tp': valid_tp
            })
            
            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f}  AUC: {train_auc:.3f}  Bal Acc: {train_bal_acc:.3f}  Sens: {train_sens:.3f}  Spec: {train_spec:.3f}")
            print(f"\t Val. Loss: {valid_loss:.3f}  AUC: {valid_auc:.3f}  Bal Acc: {valid_bal_acc:.3f}  Sens: {valid_sens:.3f}  Spec: {valid_spec:.3f}")
        else:
            train_metrics.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc})
            valid_metrics.append({'epoch': epoch + 1, 'valid_loss': valid_loss, 'valid_acc': valid_acc})
                    
            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f}  Accuracy: {train_acc}")
            print(f"\t Val. Loss: {valid_loss:.3f}  Accuracy: {valid_acc}")

        if patience_counter >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    output_dir = "/pscratch/sd/p/pakmasha/enigma_quantum/model"
    if hyperparams["model_dir"]==None:
        model.load_state_dict(torch.load(f"{output_dir}/best_time_series_model.pt", weights_only=True))
    else:
        model.load_state_dict(torch.load(f"{output_dir}/{hyperparams['model_dir']}_{hyperparams['seed']}_{tuning_set}.pt", weights_only=True))
    if hyperparams["binary"]==True:
        test_loss, test_auc, test_bal_acc, test_sens, test_spec, test_tn, test_fp, test_fn, test_tp = evaluate(model, test_loader, criterion)
        print(f"Test Loss: {test_loss:.3f}  AUC: {test_auc:.3f}  Bal Acc: {test_bal_acc:.3f}  Sens: {test_sens:.3f}  Spec: {test_spec:.3f}")
    else:
        test_loss, test_acc = evaluate_multiclass(model, test_loader, criterion)
        print(f"Test Loss: {test_loss:.3f}  Accuracy: {test_acc}")
    
    # Combine all metrics into a pandas DataFrame
    metrics = []
    for epoch in range(len(train_metrics)):
        row = {
            'epoch': epoch + 1,
            'train_loss': train_metrics[epoch]['train_loss'],
            'valid_loss': valid_metrics[epoch]['valid_loss'],
            'test_loss': test_loss,
        }
        if hyperparams["binary"]:
            row['train_auc'] = train_metrics[epoch]['train_auc']
            row['train_bal_acc'] = train_metrics[epoch]['train_bal_acc']
            row['train_sens'] = train_metrics[epoch]['train_sens']
            row['train_spec'] = train_metrics[epoch]['train_spec']
            row['train_tn'] = train_metrics[epoch]['train_tn']
            row['train_fp'] = train_metrics[epoch]['train_fp']
            row['train_fn'] = train_metrics[epoch]['train_fn']
            row['train_tp'] = train_metrics[epoch]['train_tp']
            
            row['valid_auc'] = valid_metrics[epoch]['valid_auc']
            row['valid_bal_acc'] = valid_metrics[epoch]['valid_bal_acc']
            row['valid_sens'] = valid_metrics[epoch]['valid_sens']
            row['valid_spec'] = valid_metrics[epoch]['valid_spec']
            row['valid_tn'] = valid_metrics[epoch]['valid_tn']
            row['valid_fp'] = valid_metrics[epoch]['valid_fp']
            row['valid_fn'] = valid_metrics[epoch]['valid_fn']
            row['valid_tp'] = valid_metrics[epoch]['valid_tp']
            
            row['test_auc'] = test_auc
            row['test_bal_acc'] = test_bal_acc
            row['test_sens'] = test_sens
            row['test_spec'] = test_spec
            row['test_tn'] = test_tn
            row['test_fp'] = test_fp
            row['test_fn'] = test_fn
            row['test_tp'] = test_tp
        else:
            row['train_acc'] = train_metrics[epoch]['train_acc']
            row['valid_acc'] = valid_metrics[epoch]['valid_acc']
            row['test_acc'] = test_acc
        metrics.append(row)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Save to CSV
    output_dir = "/pscratch/sd/p/pakmasha/enigma_quantum/model"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"{output_dir}/{hyperparams['target']}_{hyperparams['seed']}_{tuning_set}.csv"
    metrics_df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")
    
    if hyperparams["binary"]:
        return test_loss, test_auc, test_bal_acc, test_sens, test_spec, best_valid_loss, best_valid_auc
    else:
        return test_loss, test_acc, best_valid_loss, best_valid_acc



def seed(SEED: int) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def get_train_evaluate_class(device: torch.device, tuning_set) -> Callable:
    def train_evaluate(parameterization: dict[str, Any]) -> float:        
        print("DEBUG: Executing train_evaluate (Expects 7 return values)")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        seed(parameterization["seed"])
        
        if parameterization["dataset"] == "ABCD":
            sequence_length = 363
        elif parameterization["dataset"] == "UKB":
            sequence_length = 490
        elif parameterization["dataset"] == "ENIGMA_OCD":
            sequence_length = 115
        else:
            sequence_length = 363 # Default fallback
            
        X, y, valid_subject_ids, sites = load_fmri_data(
            parameterization["dataset"],
            parameterization["parcel_type"],
            parameterization["input_phenotype"],
            parameterization["target"],
            parameterization["input_categorical"],
        )
        
        # Prepare path for splits
        split_dir = "/pscratch/sd/p/pakmasha/enigma_quantum/splits"
        split_filename = f"{parameterization['model_dir']}_{parameterization['seed']}_{tuning_set}.txt"
        split_output_path = f"{split_dir}/{split_filename}"
        
        train_loader, val_loader, test_loader = split_and_prepare_dataloaders(
            X, y,
            valid_subject_ids,
            parameterization["batch_size"],
            sequence_length,
            device,
            parameterization["binary"],
            split_output_path,
            sites=sites if len(sites) > 0 else None
        )

        feature_dim = train_loader.dataset[0][0].shape[-1]  
        model = create_model(parameterization, device, sequence_length, feature_dim)

        init_weights(model)
        
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs")
        #     model = torch.nn.DataParallel(model)
        model = model.to(device)

        if parameterization["binary"]:
            test_loss, test_auc, test_bal_acc, test_sens, test_spec, val_loss, val_auc = train_cycle(
                model,
                parameterization,
                device,
                train_loader,
                val_loader,
                test_loader,
                tuning_set,
            )
            return test_loss, test_auc, test_bal_acc, test_sens, test_spec, val_loss, val_auc
        else:
            test_loss, test_acc, val_loss, val_acc = train_cycle(
                model,
                parameterization,
                device,
                train_loader,
                val_loader,
                test_loader,
                tuning_set,
            )
            return test_loss, test_acc, val_loss, val_acc

    return train_evaluate




if __name__ == "__main__":
    # Define Hyperparameters and Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Example Configuration for ENIGMA-OCD
    # IMPORTANT: Update "target" to the exact column name in your CSV
    config = {
        "dataset": "ENIGMA_OCD",
        "parcel_type": "Schaefer",  # Placeholder, logic uses dataset to determine path
        "input_phenotype": [],      # Use empty list to only use fMRI data
        "target": "OCD",             # CHANGE THIS to your actual target column name (e.g., diagnosis, OCD, etc.)
        "input_categorical": [],    # Keep empty if input_phenotype is empty

        # Model Hyperparams
        "qubits": 8,
        "degree": 2,
        "ansatz_layers": 2,
        "output_dim": 1,           # 1 for Binary Classification (BCEWithLogitsLoss)
        "dropout": 0.1,
        
        # Training Hyperparams
        "batch_size": 32,
        "binary": True,
        "optimizer": "Adam",
        "lr": 0.001,
        "wd": 0.001,
        "eps": 1e-8,
        "lr_sched": "cos",
        "restart_epochs": 10,
        "epochs": 50,
        "patience": 10,

        "max_grad_norm": 1.0,
        
        # Misc
        "model_dir": "QTS_ENIGMA",
        "seed": 42
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
        
    # Get the training function
    # tuning_set is a string suffix for saving files (e.g. "run1")
    tuning_suffix = "best_8qubit_2layers_MLPfc" 
    
    # --- Logging Setup ---
    log_dir = "/pscratch/sd/p/pakmasha/enigma_quantum/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/{config['model_dir']}_{config['seed']}_{tuning_suffix}.log"
    print(f"Logging to: {log_path}")

    class TeeWriter(object):
        def __init__(self, *writers):
            self.writers = writers
        def write(self, text):
            for w in self.writers:
                w.write(text)
                w.flush()
        def flush(self):
            for w in self.writers:
                w.flush()

    log_file = open(log_path, "w")
    sys.stdout = TeeWriter(sys.stdout, log_file)
    
    train_evaluate = get_train_evaluate_class(device, tuning_suffix)
    
    # Run Training
    try:
        if config["binary"]:
            loss, auc, bal_acc, sens, spec, val_loss, val_auc = train_evaluate(config)
            print(f"\nTraining Completed.")
            print(f"Final Test Loss: {loss:.4f}")
            print(f"Final Test AUC: {auc:.4f}")
            print(f"Final Balanced Accuracy: {bal_acc:.4f}")
            print(f"Final Sensitivity: {sens:.4f}")
            print(f"Final Specificity: {spec:.4f}")
        else:
            loss, metric, val_loss, val_metric = train_evaluate(config)
            print(f"\nTraining Completed.\nFinal Test Loss: {loss:.4f}\nFinal Test Accuracy: {metric:.4f}")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
