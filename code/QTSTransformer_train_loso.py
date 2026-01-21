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


def split_and_prepare_dataloaders_loso(X, y, subject_ids, sites, leave_out_site, batch_size, sequence_length, device, binary):
    """
    Split data into train, validation, and test sets using Leave-One-Site-Out Cross-Validation.
    
    Args:
        X (list of tensors): Input data for all subjects.
        y (list): Target labels.
        subject_ids (list): Subject IDs.
        sites (list): Site ID for each subject.
        leave_out_site (str): The site to use as the Test Set.
        batch_size (int): Batch size.
        sequence_length (int): Sequence length.
        device (torch.device): Device.
        binary (bool): Binary classification flag.
        
    Returns:
        train_loader, val_loader, test_loader
        train_subs, val_subs, test_subs (Lists of subjects in each split)
    """
    def create_sequences(data, labels):
        sequences, sequence_labels = [], []
        for subject_data, label in zip(data, labels):
            num_time_points = subject_data.shape[0]
            for start in range(0, num_time_points - sequence_length + 1, sequence_length):
                seq = subject_data[start:start + sequence_length]
                sequences.append(seq)
                sequence_labels.append(label)
        return sequences, sequence_labels

    y = np.array(y)
    subject_ids = np.array(subject_ids)
    sites = np.array(sites)
    
    # 1. Identify Test Set (The Left-Out Site)
    test_indices = np.where(sites == leave_out_site)[0]
    train_val_indices = np.where(sites != leave_out_site)[0]
    
    if len(test_indices) == 0:
        raise ValueError(f"No subjects found for site: {leave_out_site}")
        
    # Extract Test Data
    test_X = [X[i] for i in test_indices]
    test_y = y[test_indices]
    test_subs = subject_ids[test_indices]
    
    # Extract Train/Val Pool
    pool_X = [X[i] for i in train_val_indices]
    pool_y = y[train_val_indices]
    pool_subs = subject_ids[train_val_indices]
    pool_sites = sites[train_val_indices]
    
    # 2. Stratified Split for Train/Val
    # Create combined stratification labels (Target + Site)
    stratify_labels = [f"{lbl}_{site}" for lbl, site in zip(pool_y, pool_sites)]
    
    # Check for rare classes in the pool
    counts = Counter(stratify_labels)
    # If any group has < 2 samples, we can't stratify by it. Fallback to just Target stratification.
    if min(counts.values()) < 2:
        print(f"Warning: Falling back to target-only stratification for Train/Val split (Site: {leave_out_site} excluded)")
        stratify_labels = pool_y
        
    # 85/15 Split for Train/Val (approx similar to original 70/15/15 ratio relative to pool)
    # The original was 70/15/15 -> Train is ~82% of (Train+Val). Let's use 0.15 validation size.
    train_X_data, val_X_data, train_y, val_y, train_subs, val_subs = train_test_split(
        pool_X, pool_y, pool_subs, test_size=0.15, stratify=stratify_labels, random_state=42
    )

    # # 3. Normalization (Using Train stats only)
    # train_X_concat = torch.cat(train_X_data, dim=0)
    # train_X_mean = train_X_concat.mean(dim=0, keepdim=True)
    # train_X_std = train_X_concat.std(dim=0, keepdim=True)
    # train_X_std[train_X_std == 0] = 1e-8
    
    # def normalize_subjects(subjects, mean, std):
    #     return [(subj - mean) / std for subj in subjects]
        
    # train_X_norm = normalize_subjects(train_X_data, train_X_mean, train_X_std)
    # val_X_norm = normalize_subjects(val_X_data, train_X_mean, train_X_std)
    # test_X_norm = normalize_subjects(test_X, train_X_mean, train_X_std)
    
    # Subject-wise normalization
    def normalize_subject_wise(subjects):
        normalized = []
        for subj in subjects:
            mean = subj.mean(dim=0, keepdim=True)
            std = subj.std(dim=0, keepdim=True)
            std[std == 0] = 1e-8
            normalized.append((subj - mean) / std)
        return normalized

    train_X_norm = normalize_subject_wise(train_X_data)
    val_X_norm = normalize_subject_wise(val_X_data)
    test_X_norm = normalize_subject_wise(test_X)


    # 4. Create Sequences
    train_sequences, train_sequence_labels = create_sequences(train_X_norm, train_y)
    val_sequences, val_sequence_labels = create_sequences(val_X_norm, val_y)
    test_sequences, test_sequence_labels = create_sequences(test_X_norm, test_y)

    # 5. Create DataLoaders
    def create_dataloader(sequences, labels):
        x_tensors = [seq.to(device) for seq in sequences]
        y_tensor = torch.tensor(labels, dtype=torch.float32 if binary else torch.long, device=device)        
        if len(x_tensors) == 0:
            return None
        dataset = TensorDataset(torch.stack(x_tensors), y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_dataloader(train_sequences, train_sequence_labels)
    val_loader = create_dataloader(val_sequences, val_sequence_labels)
    test_loader = create_dataloader(test_sequences, test_sequence_labels)
    
    # Save splits to file
    split_dir = "/pscratch/sd/p/pakmasha/enigma_quantum/splits"
    os.makedirs(split_dir, exist_ok=True)
    split_filename = f"{split_dir}/loso_split_{leave_out_site}.txt"
    
    with open(split_filename, "w") as f:
        f.write("train_subjects\n")
        np.savetxt(f, train_subs, fmt="%s")
        f.write("val_subjects\n")
        np.savetxt(f, val_subs, fmt="%s")
        f.write("test_subjects\n")
        np.savetxt(f, test_subs, fmt="%s")
    
    print(f"Splits saved to {split_filename}")
    
    return train_loader, val_loader, test_loader, train_subs, val_subs, test_subs


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
    y_true, y_scores = [], []
    
    for x, y in tqdm(iterator, leave=False, desc="Training"):
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
        
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            yhat = torch.sigmoid(yhat)
            
        y_true.extend(y.cpu().numpy().astype(int))
        y_scores.extend(yhat.detach().cpu().squeeze().numpy())
    
    try:
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        if len(np.unique(y_true)) != 2:
            auc = 0.5 # Default if only one class in batch
        else:
            auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = 0.5
        
    return epoch_loss / len(iterator), auc


def evaluate(
    model: torch.nn.Module,
    iterator: DataLoader,
    criterion,
):
    model.eval()
    epoch_loss = 0
    y_true, y_scores = [], []
    
    with torch.no_grad():
        for x, y in tqdm(iterator, leave=False, desc="Evaluating"):
            yhat = model(x)
            loss = criterion(yhat.squeeze(), y)
            epoch_loss += loss.item()
            
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                yhat = torch.sigmoid(yhat)
                
            y_true.extend(y.cpu().numpy().astype(int))
            y_scores.extend(yhat.cpu().squeeze().numpy())
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Metrics
    try:
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_scores)
            preds = (y_scores >= 0.5).astype(int)
            bal_acc = balanced_accuracy_score(y_true, preds)
            sens = recall_score(y_true, preds, pos_label=1, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            auc = 0.5
            bal_acc = 0.5
            sens = 0.0
            spec = 0.0
            tn, fp, fn, tp = 0, 0, 0, 0
    except ValueError as e:
        print(f"Metrics Error: {e}")
        auc, bal_acc, sens, spec = 0.5, 0.5, 0, 0
        tn, fp, fn, tp = 0, 0, 0, 0
        
    return epoch_loss / len(iterator), auc, bal_acc, sens, spec, y_true, y_scores


def train_cycle(
    model: torch.nn.Module,
    hyperparams: dict[str, Any],
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    fold_name
):
    if hyperparams["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"], eps=hyperparams["eps"]
        )
    elif hyperparams["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
        model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"], eps=hyperparams["eps"]
        )    
        
    scheduler = None
    if hyperparams["lr_sched"] == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=hyperparams["restart_epochs"]
        )

    criterion = torch.nn.BCEWithLogitsLoss()

    best_valid_loss = float("inf")
    patience = hyperparams.get("patience", 10)
    patience_counter = 0
    
    # Helper to save best model for this fold
    fold_model_path = f"/pscratch/sd/p/pakmasha/enigma_quantum/model/loso_fold_{fold_name}.pt"
    os.makedirs(os.path.dirname(fold_model_path), exist_ok=True)

    for epoch in range(hyperparams["epochs"]):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, hyperparams["max_grad_norm"], scheduler)
        valid_loss, valid_auc, _, _, _, _, _ = evaluate(model, val_loader, criterion)
        
        print(f"Fold {fold_name} | Epoch {epoch+1} | T.Loss: {train_loss:.4f} | T.AUC: {train_auc:.4f} | V.Loss: {valid_loss:.4f} | V.AUC: {valid_auc:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), fold_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} for fold {fold_name}")
            break

    # Load best model and evaluate on Test Set (Left-out Site)
    model.load_state_dict(torch.load(fold_model_path))
    test_loss, test_auc, test_bal_acc, test_sens, test_spec, test_y_true, test_y_probs = evaluate(model, test_loader, criterion)
    
    return {
        "loss": test_loss,
        "auc": test_auc,
        "bal_acc": test_bal_acc,
        "sens": test_sens,
        "spec": test_spec,
        "y_true": test_y_true,
        "y_probs": test_y_probs
    }


def seed(SEED: int) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def run_loso_cv(hyperparams, device, tuning_suffix):
    """
    Main Loop for Leave-One-Site-Out Cross-Validation
    """
    seed(hyperparams["seed"])
    
    # 1. Load Data
    print("Loading Data...")
    X, y, subject_ids, sites = load_fmri_data(
        hyperparams["dataset"],
        hyperparams["parcel_type"],
        hyperparams["input_phenotype"],
        hyperparams["target"],
        hyperparams["input_categorical"],
    )
    
    unique_sites = list(np.unique(sites))
    print(f"Found {len(unique_sites)} unique sites: {unique_sites}")
    
    # Metrics Storage
    global_y_true = []
    global_y_probs = []
    site_results = []
    
    # Output File Setup
    output_dir = "/pscratch/sd/p/pakmasha/enigma_quantum/loso_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. LOSO Loop
    for site in unique_sites:
        print(f"\n--- Processing Site: {site} ---")
        
        # Skip site if too small (e.g., just 1 subject, can't calc metrics anyway mostly)
        site_indices = [i for i, s in enumerate(sites) if s == site]
        if len(site_indices) < 2:
            print(f"Skipping site {site} due to insufficient samples ({len(site_indices)})")
            continue

        # Prepare Data
        # sequence_length logic
        if hyperparams["dataset"] == "ABCD": seq_len = 363
        elif hyperparams["dataset"] == "UKB": seq_len = 490
        elif hyperparams["dataset"] == "ENIGMA_OCD": seq_len = 115
        else: seq_len = 363
            
        try:
            train_loader, val_loader, test_loader, train_subs, val_subs, test_subs = split_and_prepare_dataloaders_loso(
                X, y, subject_ids, sites, 
                leave_out_site=site,
                batch_size=hyperparams["batch_size"],
                sequence_length=seq_len,
                device=device,
                binary=hyperparams["binary"]
            )
        except ValueError as e:
            print(f"Splitting error for site {site}: {e}")
            continue

        if train_loader is None or test_loader is None:
            print("Skipping site due to empty loader.")
            continue

        # Create Model
        feature_dim = train_loader.dataset[0][0].shape[-1]
        model = create_model(hyperparams, device, seq_len, feature_dim)
        init_weights(model)
        model = model.to(device)
        
        # Train
        fold_res = train_cycle(
            model, hyperparams, device, train_loader, val_loader, test_loader, fold_name=site
        )
        
        # Store Results
        print(f"Site {site} Results: AUC={fold_res['auc']:.3f}, BalAcc={fold_res['bal_acc']:.3f}")
        
        site_metrics = {
            "site": site,
            "n_samples": len(fold_res["y_true"]),
            "auc": fold_res["auc"],
            "bal_acc": fold_res["bal_acc"],
            "sens": fold_res["sens"],
            "spec": fold_res["spec"],
            "loss": fold_res["loss"]
        }
        site_results.append(site_metrics)
        
        global_y_true.extend(fold_res["y_true"])
        global_y_probs.extend(fold_res["y_probs"])
        
        # Save intermediate per-site results
        pd.DataFrame(site_results).to_csv(f"{output_dir}/loso_site_metrics_{tuning_suffix}.csv", index=False)

    # 3. Final Evaluation
    print("\n--- Final LOSO Evaluation ---")
    
    # Global Pooling Metrics
    global_y_true = np.array(global_y_true)
    global_y_probs = np.array(global_y_probs)
    
    global_auc = roc_auc_score(global_y_true, global_y_probs)
    global_preds = (global_y_probs >= 0.5).astype(int)
    global_bal_acc = balanced_accuracy_score(global_y_true, global_preds)
    global_sens = recall_score(global_y_true, global_preds, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(global_y_true, global_preds, labels=[0, 1]).ravel()
    global_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print(f"Global Pooling Results:")
    print(f"  AUC: {global_auc:.4f}")
    print(f"  Balanced Accuracy: {global_bal_acc:.4f}")
    print(f"  Sensitivity: {global_sens:.4f}")
    print(f"  Specificity: {global_spec:.4f}")
    
    # Macro-Averaging
    df_sites = pd.DataFrame(site_results)
    macro_auc_mean = df_sites["auc"].mean()
    macro_auc_std = df_sites["auc"].std()
    
    print(f"Macro-Averaging Results (Mean across sites):")
    print(f"  Mean AUC: {macro_auc_mean:.4f} ± {macro_auc_std:.4f}")
    
    # Save Final Summary
    summary = {
        "Metric": ["Global_AUC", "Global_BalAcc", "Global_Sens", "Global_Spec", "Macro_Mean_AUC", "Macro_Std_AUC"],
        "Value": [global_auc, global_bal_acc, global_sens, global_spec, macro_auc_mean, macro_auc_std]
    }
    pd.DataFrame(summary).to_csv(f"{output_dir}/loso_summary_{tuning_suffix}.csv", index=False)
    print(f"Results saved to {output_dir}")



if __name__ == "__main__":
    # Define Hyperparameters and Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    config = {
        "dataset": "ENIGMA_OCD",
        "parcel_type": "Schaefer",
        "input_phenotype": [], 
        "target": "OCD",
        "input_categorical": [],

        # Model Hyperparams
        "qubits": 8,
        "degree": 2,
        "ansatz_layers": 2,
        "output_dim": 1,
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
        "rotation_scale": 1.0,
        
        # Misc
        "model_dir": "QTS_ENIGMA_LOSO",
        "seed": 42
    }
    
    tuning_suffix = "loso_run_v1" 
    
    # --- Logging Setup ---
    log_dir = "/pscratch/sd/p/pakmasha/enigma_quantum/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/LOSO_{config['target']}_{config['seed']}_{tuning_suffix}.log"
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
    
    # Run
    run_loso_cv(config, device, tuning_suffix)
