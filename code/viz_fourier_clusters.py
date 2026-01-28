# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Try to import UMAP
try:
    import umap
    HAS_UMAP = True
    print("UMAP library found. Will use UMAP for visualization.")
except ImportError:
    HAS_UMAP = False
    print("UMAP library not found. Skipping UMAP.")

# %%
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Paths
ROI0_MATRIX_PATH = "/pscratch/sd/p/pakmasha/enigma_quantum/data/pca_matrix_roi0.npy"
ALL_ROIS_MATRIX_PATH = "/pscratch/sd/p/pakmasha/enigma_quantum/data/pca_matrix_all_rois.npy"

METADATA_PATH = "/pscratch/sd/p/pakmasha/enigma_quantum/metadata/ENIGMA_QC_final_subject_list.csv"
PARAMS_PATH = "/pscratch/sd/p/pakmasha/enigma_quantum/metadata/ENIGMA_OCD_parameters.csv"

# Output Directories
BASE_OUTPUT_DIR = "/pscratch/sd/p/pakmasha/enigma_quantum/logs/clustering_exploration"
ROI0_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "ft_roi0")
ALL_ROIS_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "ft_all_rois")

# Ensure output directories exist
os.makedirs(ROI0_OUTPUT_DIR, exist_ok=True)
os.makedirs(ALL_ROIS_OUTPUT_DIR, exist_ok=True)

# Columns mapping
TARGET_COL = "OCD" 
SITE_COL = "Sample"
SUBJECT_ID_COL = "Unique_ID"

# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------

def load_and_match_metadata(matrix_path, metadata_path, params_path, tr_limit=2.0):
    """
    Loads the frequency matrix and matches it with the metadata.
    This reconstructs the subject filtering logic to ensure metadata lines up with the matrix rows.
    """
    print(f"Loading matrix from: {matrix_path}")
    data_matrix = np.load(matrix_path)
    
    # We need to reconstruct the list of subjects to match rows to metadata
    # IMPORTANT: This MUST match the exact logic used in fourier_transform_demo.py
    # Re-importing TR function logic briefly or using the same list if saved (but we didn't save IDs order)
    # Solution: We will scan the folders again using the SAME logic to get the ID order.
    
    # --- Recreating the subject order ---
    import glob
    DATA_ROOT = '/pscratch/sd/p/pakmasha/enigma_quantum_data_before_resampling/enigma_quantum_data'
    
    def repetition_time(site):
        # Simplified lookup logic copy-pasted/imported to ensure consistency
        # Assuming the standard list
    
        if 'Amsterdam-AMC' in site: TR = 2.375
        elif 'Amsterdam-VUmc' in site: TR = 1.8
        elif 'Barcelona-HCPB' in site: TR = 2
        elif 'Bergen' in site: TR = 1.8
        elif 'Braga-UMinho-Braga-1.5T' in site: TR = 2
        elif 'Braga-UMinho-Braga-1.5T-act' in site: TR = 2
        elif 'Braga-UMinho-Braga-3T' in site: TR = 1
        elif 'Brazil' in site: TR = 2
        elif 'Cape-Town-UCT-Allegra' in site: TR = 1.6
        elif 'Cape-Town-UCT-Skyra' in site: TR = 1.73
        elif 'Chiba-CHB' in site: TR = 2.3
        elif 'Chiba-CHBC' in site: TR = 2.3 
        elif 'Chiba-CHBSRPB' in site: TR = 2.5 
        elif 'Dresden' in site: TR = 0.8 
        elif 'Kyoto-KPU-Kyoto1.5T' in site: TR = 2.411 
        elif 'Kyoto-KPU-Kyoto3T' in site: TR = 2
        elif 'Kyushu' in site: TR = 2.5
        elif 'Milan-HSR' in site: TR = 2
        elif 'New-York' in site: TR = 1
        elif 'NYSPI-Columbia-Adults' in site: TR = 0.85
        elif 'NYSPI-Columbia-Pediatric' in site: TR = 0.85
        elif 'Yale-Pittinger-HCP-Prisma' in site: TR = 0.8
        elif 'Yale-Pittinger-HCP-Trio' in site: TR = 0.7
        elif 'Yale-Pittinger-Yale-2014' in site: TR = 2
        elif 'Bangalore-NIMHANS' in site: TR = 2 
        elif 'Barcelone-Bellvitge-ANTIGA-1.5T' in site: TR = 2
        elif 'Barcelone-Bellvitge-COMPULSE-3T' in site: TR = 2
        elif 'Barcelone-Bellvitge-PROV-1.5T' in site: TR = 2
        elif 'Barcelone-Bellvitge-RESP-CBT-3T' in site: TR = 2
        elif 'Seoul-SNU' in site: TR = 3.5
        elif 'Shanghai-SMCH' in site: TR = 3
        elif 'UCLA' in site: TR = 2
        elif 'Vancouver-BCCHR' in site: TR = 2
        elif 'Yale-Gruner' in site: TR = 2
        else: return None
        return TR
        
    all_subject_paths = glob.glob(os.path.join(DATA_ROOT, "*", "*.npy"))
    valid_ids = []
    
    # Sort paths if the original script relied on glob order (glob is usually OS dependent but consistent per run)
    # However, to be safe, we hope the order hasn't changed. 
    # Ideally we would have saved the ID list. We will assume glob order is stable.
    
    for path in all_subject_paths:
        subject_id = os.path.basename(os.path.dirname(path))
        tr = repetition_time(subject_id)
        if tr is not None and tr <= tr_limit:
            valid_ids.append(subject_id)
            
    if len(valid_ids) != data_matrix.shape[0]:
        print(f"WARNING: IDs count ({len(valid_ids)}) != Matrix rows ({data_matrix.shape[0]}).")
        print("This might be due to glob order changes or file changes.")
        # We will proceed but this is risky without a saved ID list.
    
    # --- Load Metadata ---
    df_meta = pd.read_csv(metadata_path)
    
    # Filter metadata to keep only the subjects in our matrix, in the correct order
    # We create a dataframe from our valid_ids list
    df_matrix_order = pd.DataFrame({SUBJECT_ID_COL: valid_ids})
    
    # Merge to attach metadata to our specific matrix order
    # 'how=left' preserves the matrix order (left side)
    df_merged = df_matrix_order.merge(df_meta, on=SUBJECT_ID_COL, how='left')
    
    # --- Load Parameters (Scanner/Model) ---
    if os.path.exists(params_path):
        df_params = pd.read_csv(params_path)
        if 'Site' in df_params.columns:
            df_params_clean = df_params.drop_duplicates(subset=['Site'])[['Site', 'Scanner', 'Model_unified']]
            # Merge on Site
            df_merged = df_merged.merge(df_params_clean, left_on=SITE_COL, right_on='Site', how='left')
    
    return data_matrix, df_merged

def run_exploration(df, data_matrix, output_dir, method='UMAP', params_list=[], n_components=2):
    """
    Runs dimensionality reduction and saves plots.
    """
    if method == 'UMAP' and not HAS_UMAP:
        return

    # 3D support
    is_3d = (n_components == 3)

    for params in params_list:
        print(f"Running {method} ({n_components}D): {params['tag']}")
        
        kwargs = {k: v for k, v in params.items() if k != 'tag'}
        
        try:
            if method == 'UMAP':
                reducer = umap.UMAP(n_components=n_components, random_state=42, **kwargs)
                embedding = reducer.fit_transform(data_matrix)
            else:
                tsne = TSNE(n_components=n_components, random_state=42, init='pca', learning_rate='auto', **kwargs)
                embedding = tsne.fit_transform(data_matrix)

            # Prepare plotting df
            df_plot = df.copy()
            df_plot['x_emb'] = embedding[:, 0]
            df_plot['y_emb'] = embedding[:, 1]
            if is_3d:
                df_plot['z_emb'] = embedding[:, 2]
            
            # Create Plot Grid
            fig = plt.figure(figsize=(24, 6))
            fig.suptitle(f"{method} {n_components}D - {params['tag']}\nParams: {kwargs}", fontsize=16)
            
            hue_columns = [
                (SITE_COL, "By Site"),
                (TARGET_COL, "By Diagnosis"),
                ('Scanner', "By Scanner"),
                ('Model_unified', "By Model")
            ]

            for i, (col, title) in enumerate(hue_columns):
                if is_3d:
                    ax = fig.add_subplot(1, 4, i+1, projection='3d')
                else:
                    ax = fig.add_subplot(1, 4, i+1)

                if col in df_plot.columns:
                    unique_vals = df_plot[col].unique()
                    # Handle NaNs
                    unique_vals = [v for v in unique_vals if pd.notna(v)]
                    n_colors = len(unique_vals)
                    palette_name = 'coolwarm' if col == TARGET_COL else 'tab20'
                    
                    if is_3d:
                        # 3D Plotting
                        colors = sns.color_palette(palette_name, n_colors) if n_colors > 0 else []
                        color_map = dict(zip(unique_vals, colors))
                        
                        for val in unique_vals:
                            subset = df_plot[df_plot[col] == val]
                            ax.scatter(subset['x_emb'], subset['y_emb'], subset['z_emb'], 
                                     label=str(val), s=15, alpha=0.6,
                                     c=[color_map[val]])
                        
                        ax.set_xlabel('Dim 1')
                        ax.set_ylabel('Dim 2')
                        ax.set_zlabel('Dim 3')
                        
                        # Legend outside
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small', markerscale=2)

                    else:
                        # 2D Plotting
                        sns.scatterplot(data=df_plot, x='x_emb', y='y_emb', hue=col, 
                                        palette=palette_name, s=20, alpha=0.6, ax=ax, 
                                        legend='full') # Always show full legend
                        
                        # Move legend outside
                        # For 'Site', reducing font size or using columns helps, but placing outside is safest
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')

                        ax.set_xlabel('Dim 1')
                        ax.set_ylabel('Dim 2')

                    ax.set_title(title)
                else:
                    ax.text(0.5, 0.5, f"{col} missing", ha='center')

            plt.tight_layout()
            # Adjust layout to make room for external legends
            plt.subplots_adjust(right=0.9) 

            
            # Save
            filename = f"{method}_{n_components}D_{params['tag'].replace(' ', '_')}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150)
            print(f"Saved: {filepath}")
            plt.close()
            
        except Exception as e:
            print(f"Error in {params['tag']}: {e}")

# %%
# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":
    
    # --- Define Hyperparam Grids ---
    
    # UMAP Grid
    umap_params = []
    for n in [5, 15, 30, 50, 100]:
        for d in [0.0, 0.1, 0.25]:
            for m in ['cosine', 'euclidean']:
                umap_params.append({'n_neighbors': n, 'min_dist': d, 'metric': m, 'tag': f"N{n}_D{d}_{m}"})
                
    # t-SNE Grid
    tsne_params = []
    for p in [5, 15, 30, 50, 100]:
        for m in ['cosine', 'euclidean']:
            tsne_params.append({'perplexity': p, 'metric': m, 'tag': f"Perp{p}_{m}"})

    # --- 1. Process ROI 0 ---
    print("\n\n================================================")
    print("Processing ROI 0 Matrix...")
    print("================================================")
    
    X_roi0, df_roi0 = load_and_match_metadata(ROI0_MATRIX_PATH, METADATA_PATH, PARAMS_PATH)
    
    # Check for NaNs
    if np.isnan(X_roi0).any():
        print("Creating NaN mask and imputing with 0...")
        X_roi0 = np.nan_to_num(X_roi0)
        
    # Scale Data
    X_roi0_scaled = StandardScaler().fit_transform(X_roi0)
    
    # Run UMAP (2D & 3D)
    # run_exploration(df_roi0, X_roi0_scaled, ROI0_OUTPUT_DIR, 'UMAP', umap_params, n_components=2)
    # run_exploration(df_roi0, X_roi0_scaled, ROI0_OUTPUT_DIR, 'UMAP', umap_params, n_components=3)
    
    # Run t-SNE (2D & 3D) - Use PCA reduction first for speed
    print("Pre-computing PCA for t-SNE...")
    X_roi0_pca = PCA(n_components=13).fit_transform(X_roi0_scaled)
    
    run_exploration(df_roi0, X_roi0_pca, ROI0_OUTPUT_DIR, 't-SNE', tsne_params, n_components=2)
    run_exploration(df_roi0, X_roi0_pca, ROI0_OUTPUT_DIR, 't-SNE', tsne_params, n_components=3)


    # --- 2. Process ALL ROIs ---
    print("\n\n================================================")
    print("Processing ALL ROIs Matrix...")
    print("================================================")
    
    X_all, df_all = load_and_match_metadata(ALL_ROIS_MATRIX_PATH, METADATA_PATH, PARAMS_PATH)
    
    # Check for NaNs
    if np.isnan(X_all).any():
        print("Creating NaN mask and imputing with 0...")
        X_all = np.nan_to_num(X_all)
        
    # Scale Data
    # Important: Scaling a huge matrix (2000 subjs x 4000 dims) is heavy.
    # Be mindful of memory here.
    print("Standard Scaling big matrix...")
    X_all_scaled = StandardScaler().fit_transform(X_all)
    
    # Run UMAP (2D only to save time, or match ROI0 logic? Let's do both but reduced grid if needed)
    # Using full grid as requested
    run_exploration(df_all, X_all_scaled, ALL_ROIS_OUTPUT_DIR, 'UMAP', umap_params, n_components=2)
    run_exploration(df_all, X_all_scaled, ALL_ROIS_OUTPUT_DIR, 'UMAP', umap_params, n_components=3)
    
    # Run t-SNE
    print("Pre-computing PCA for t-SNE...")
    X_all_pca = PCA(n_components=88).fit_transform(X_all_scaled)
    
    run_exploration(df_all, X_all_pca, ALL_ROIS_OUTPUT_DIR, 't-SNE', tsne_params, n_components=2)
    run_exploration(df_all, X_all_pca, ALL_ROIS_OUTPUT_DIR, 't-SNE', tsne_params, n_components=3)

    print("\nDone! All plots saved.")
