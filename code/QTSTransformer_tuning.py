import os
import sys
import math
import pandas as pd
import numpy as np
import torch
import optuna
from datetime import datetime

# Add the current directory to path so we can import the train module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from QTSTransformer_train import get_train_evaluate_class
except ImportError:
    # If running from project root
    sys.path.append("code/model_code")
    from QTSTransformer_train import get_train_evaluate_class

# Global timestamp to group all trials in this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def objective(trial):
    """
    Optuna Objective Function.
    suggests hyperparameters -> runs training -> returns Validation AUC.
    """
    
    # ---------------------------------------------------------
    # 1. Hyperparameter Search Space
    # ---------------------------------------------------------
    # Architecture
    qubits = trial.suggest_categorical("qubits", [4, 6, 8])
    degree = trial.suggest_categorical("degree", [1, 2])
    ansatz_layers = trial.suggest_categorical("ansatz_layers", [1, 2])
    
    # Rotation Scale
    # We use a categorical suggestion to pick between specific mathematical constants
    # that are relevant for quantum rotations (1.0 vs Pi vs 2Pi)
    # rot_scale_choice = trial.suggest_categorical("rotation_scale_str", ["1.0", "pi", "2pi"])
    # if rot_scale_choice == "1.0":
    #     rotation_scale = 1.0
    # elif rot_scale_choice == "pi":
    #     rotation_scale = math.pi
    # else:
    #     rotation_scale = 2 * math.pi
        
    # Regularization
    dropout = trial.suggest_categorical("dropout", [0.1, 0.3, 0.5])
    wd = trial.suggest_categorical("wd", [1e-4, 1e-3, 1e-2])
    
    # Training
    lr = trial.suggest_categorical("lr", [1e-3, 5e-4])
    
    # ---------------------------------------------------------
    # 1b. Duplicate Check
    # ---------------------------------------------------------
    # TPE can sometimes get stuck resampling the best parameters. 
    # We check if these params have been evaluated to save time.
    for t in trial.study.trials:
        # Check against completed trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.params == trial.params:
            print(f"\n--- Trial {trial.number} is a duplicate of Trial {t.number} ---")
            print("Skipping training and returning previous value.")
            
            # Copy metrics to current trial for consistency in logs/analysis
            for k, v in t.user_attrs.items():
                trial.set_user_attr(k, v)
                
            return t.value
    
    
    # ---------------------------------------------------------
    # 2. Configuration Dictionary
    # ---------------------------------------------------------
    params = {
        # Search Params
        "qubits": qubits,
        "degree": degree,
        "ansatz_layers": ansatz_layers,
        "rotation_scale": 1.0,
        "dropout": dropout,
        "wd": wd,
        "lr": lr,
        
        # Fixed Params
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "restart_epochs": 10,
        "max_grad_norm": 1.0,
        "optimizer": "Adam",
        "lr_sched": "cos",
        "eps": 1e-8,
        "binary": True,
        "output_dim": 1,
        "dataset": "ENIGMA_OCD",
        "parcel_type": "Schaefer", # Logic inside train script handles path construction based on dataset
        "input_phenotype": [],
        "target": "OCD",
        "input_categorical": [],
        "model_dir": "QTS_TUNING",
        "seed": 42
    }

    # ---------------------------------------------------------
    # 3. Execution
    # ---------------------------------------------------------
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a unique tuning ID for file naming (checkpoints/logs)
    tuning_set = f"trial_{trial.number}_{TIMESTAMP}"
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: {params}")

    try:
        # Get the training function configured for this run
        train_func = get_train_evaluate_class(device, tuning_set)
        
        # Run Training
        result = train_func(params)
        
        # Validate Return Values and Unpack
        if len(result) == 7:
            test_loss, test_auc, test_bal_acc, test_sens, test_spec, val_loss, val_auc = result
        else:
            print(f"Error: train_func returned {len(result)} values (expected 7).")
            print(f"Returned values: {result}")
            print(f"Debug: QTSTransformer_train imported from {sys.modules['QTSTransformer_train'].__file__}")
            raise ValueError(f"not enough values to unpack (expected 7, got {len(result)})")
            
        # Store detailed metrics in trial user_attrs for analysis later
        trial.set_user_attr("test_loss", test_loss)
        trial.set_user_attr("test_auc", test_auc)
        trial.set_user_attr("test_bal_acc", test_bal_acc)
        trial.set_user_attr("test_sens", test_sens)
        trial.set_user_attr("test_spec", test_spec)
        trial.set_user_attr("val_loss", val_loss)
        trial.set_user_attr("val_auc", val_auc)
        
        # We maximize Validation AUC
        return val_auc

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a poor score to penalize failure (or could raise optuna.TrialPruned if using pruning)
        return 0.0


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


def main():
    # Setup Logging
    log_dir = "/pscratch/sd/p/pakmasha/enigma_quantum/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/tuning_{TIMESTAMP}.log"
    
    log_file = open(log_path, "w")
    sys.stdout = TeeWriter(sys.stdout, log_file)
    sys.stderr = TeeWriter(sys.stderr, log_file)
    
    print(f"Logging to: {log_path}")

    # 4. Explicitly add StreamHandler to Optuna logger using the open log_file
    # This avoids potential file locking conflicts on Windows and ensures logs go to the same stream
    import logging
    optuna_logger = optuna.logging.get_logger("optuna")
    
    # Handler 1: Main log file (shared with stdout/stderr)
    optuna_handler = logging.StreamHandler(stream=log_file) 
    optuna_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    optuna_handler.setLevel(logging.INFO)
    optuna_logger.addHandler(optuna_handler)
    
    # Handler 2: Clean tracking log (Optuna only)
    # This separate file will contain ONLY the trial results/optuna logs, not the verbose training output
    clean_log_path = f"{log_dir}/qts_enigma_tuning_{TIMESTAMP}_tracking.txt"
    clean_handler = logging.FileHandler(clean_log_path)
    clean_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    clean_handler.setLevel(logging.INFO)
    optuna_logger.addHandler(clean_handler)
    
    print(f"Tracking clean results to: {clean_log_path}")

    # Ensure optuna verbosity is at least INFO
    optuna.logging.set_verbosity(optuna.logging.INFO)

    print(f"Initializing Optuna Study... (Timestamp: {TIMESTAMP})")
    
    # Create output directory
    output_dir = "/pscratch/sd/p/pakmasha/enigma_quantum/tuning_results"
    os.makedirs(output_dir, exist_ok=True)
    
    storage_name = f"sqlite:///{output_dir}/qts_enigma_tuning_{TIMESTAMP}.db"
    
    # Create Study
    # Direction is 'maximize' because we return Validation AUC
    study = optuna.create_study(
        direction="maximize",
        study_name=f"QTS_Enigma_Tuning_{TIMESTAMP}",
        storage=storage_name,   # Save progress to DB file
        load_if_exists=True
    )
    
    print(f"Study created. Storage: {storage_name}")
    print("Starting optimization...")
    
    # Run Optimization
    # n_trials can be adjusted. 30-50 is a good start for this search space.
    study.optimize(objective, n_trials=50)
    
    print("\n--- Hyperparameter Tuning Completed ---")
    
    # Best Result
    if len(study.trials) > 0:
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print(f"  Value (Val AUC): {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
        # Save summary to CSV
        df = study.trials_dataframe()
        csv_file = f"{output_dir}/optuna_results_{TIMESTAMP}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Summary CSV saved to: {csv_file}")
    else:
        print("No trials completed.")

if __name__ == "__main__":
    main()
