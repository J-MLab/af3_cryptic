#!/usr/bin/env python3
"""
Script to analyze pose RMSDs with bootstrapping.
For each PDB case, randomly samples 5, 10, 20, ..., 100 models (keeping all 5 samples per seed together)
and plots average RMSD vs number of seeds sampled.

Now creates 3 plots:
1. Pose RMSD (from original data)
2. Open RMSD (AF3 nolig vs AF3 lig)
3. Closed RMSD (AF3 nolig vs AF3 lig)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

def parse_seed_from_pdb_name(pdb_name):
    """Extract seed number from pdb_seed column (e.g., '1alb_seed-90_sample-4' -> 90)"""
    match = re.search(r'seed-(\d+)', pdb_name)
    if match:
        return int(match.group(1))
    return None

def bootstrap_rmsd_analysis_structs(df, n_seeds, n_bootstrap=100):
    """
    Randomly sample n_seeds (keeping all 5 samples per seed together) and calculate average RMSD.
    Repeat n_bootstrap times and return mean and std of average RMSDs.
    """
    # Group by seed
    unique_seeds = df['seed'].unique()
    
    if n_seeds > len(unique_seeds):
        return None, None
    
    avg_rmsds = []
    for _ in range(n_bootstrap):
        # Randomly sample seeds
        sampled_seeds = np.random.choice(unique_seeds, size=n_seeds, replace=True)
        
        # Get all rows for these seeds
        sampled_rows = df[df['seed'].isin(sampled_seeds)]
        
        # Calculate average RMSD
        avg_rmsd = sampled_rows['loop_rmsd'].mean()
        avg_rmsds.append(avg_rmsd)
    
    return np.mean(avg_rmsds), np.std(avg_rmsds)

def bootstrap_rmsd_analysis(df, n_seeds, n_bootstrap=100):
    """
    Randomly sample n_seeds (keeping all 5 samples per seed together) and calculate average RMSD.
    Repeat n_bootstrap times and return mean and std of average RMSDs.
    """
    # Group by seed
    unique_seeds = df['seed'].unique()
    
    if n_seeds > len(unique_seeds):
        return None, None
    
    avg_rmsds = []
    for _ in range(n_bootstrap):
        # Randomly sample seeds
        sampled_seeds = np.random.choice(unique_seeds, size=n_seeds, replace=True)
        
        # Get all rows for these seeds
        sampled_rows = df[df['seed'].isin(sampled_seeds)]
        
        # Calculate average RMSD
        avg_rmsd = sampled_rows['lig_RMSD'].mean()
        avg_rmsds.append(avg_rmsd)
    
    return np.mean(avg_rmsds), np.std(avg_rmsds)

def analyze_pdb_case(csv_file, n_bootstrap=100):
    """
    Analyze a single PDB case file.
    Returns arrays of seed counts, mean RMSDs, and std RMSDs.
    """
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Extract seed numbers
    df['seed'] = df['pdb_seed'].apply(parse_seed_from_pdb_name)
    
    # Remove any rows where seed couldn't be parsed
    df = df[df['seed'].notna()]
    
    # Determine maximum number of seeds
    n_unique_seeds = df['seed'].nunique()
    
    # Sample sizes: 5, 10, 20, ..., up to 100 or max available
    seed_counts = list(range(5, min(105, n_unique_seeds + 1), 5))
    if 10 not in seed_counts and n_unique_seeds >= 10:
        seed_counts = sorted([5, 10] + [x for x in seed_counts if x not in [5, 10]])
    
    mean_rmsds = []
    std_rmsds = []
    
    for n_seeds in seed_counts:
        mean_rmsd, std_rmsd = bootstrap_rmsd_analysis(df, n_seeds, n_bootstrap)
        if mean_rmsd is not None:
            mean_rmsds.append(mean_rmsd)
            std_rmsds.append(std_rmsd)
        else:
            break
    
    # Trim to actual computed values
    seed_counts = seed_counts[:len(mean_rmsds)]
    
    return seed_counts, mean_rmsds, std_rmsds

def analyze_pdb_case_structs(csv_file, n_bootstrap=100):
    """
    Analyze a single PDB case file.
    Returns arrays of seed counts, mean RMSDs, and std RMSDs.
    """
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Extract seed numbers
    df['seed'] = df['model'].apply(parse_seed_from_pdb_name)
    
    # Remove any rows where seed couldn't be parsed
    df = df[df['seed'].notna()]
    
    # Determine maximum number of seeds
    n_unique_seeds = df['seed'].nunique()
    
    # Sample sizes: 5, 10, 20, ..., up to 100 or max available
    seed_counts = list(range(5, min(105, n_unique_seeds + 1), 5))
    if 10 not in seed_counts and n_unique_seeds >= 10:
        seed_counts = sorted([5, 10] + [x for x in seed_counts if x not in [5, 10]])
    
    mean_rmsds = []
    std_rmsds = []
    
    for n_seeds in seed_counts:
        mean_rmsd, std_rmsd = bootstrap_rmsd_analysis_structs(df, n_seeds, n_bootstrap)
        if mean_rmsd is not None:
            mean_rmsds.append(mean_rmsd)
            std_rmsds.append(std_rmsd)
        else:
            break
    
    # Trim to actual computed values
    seed_counts = seed_counts[:len(mean_rmsds)]
    
    return seed_counts, mean_rmsds, std_rmsds

def plot_pose_rmsd(csv_files, names, af_pdb):
    """Create plot for pose RMSD (original data)"""
    n_cases = min(len(csv_files), 16)
    n_rows = 4
    n_cols = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    axes = axes.flatten()
    
    # Process each case
    for idx, csv_file in enumerate(csv_files[:16]):
        # Extract PDB ID from filename
        pdb_id = csv_file.stem.replace('_lig_RMSDs', '')
        
        print(f"Processing pose RMSD for {pdb_id}...")
        
        index = af_pdb.index(pdb_id.upper()) if pdb_id.upper() in af_pdb else None
        
        try:
            seed_counts, mean_rmsds, std_rmsds = analyze_pdb_case(csv_file, n_bootstrap=100)
            
            ax = axes[idx]
            
            # Convert to numpy arrays for easier calculation
            seed_counts_arr = np.array(seed_counts)
            mean_rmsds_arr = np.array(mean_rmsds)
            std_rmsds_arr = np.array(std_rmsds)
            
            # Plot mean line
            ax.plot(seed_counts_arr, mean_rmsds_arr, 
                   marker='o', linestyle='-', markersize=4, linewidth=1.5,
                   color='#1f77b4', zorder=3)
            
            # Add shaded region for ±1 standard deviation
            ax.fill_between(seed_counts_arr, 
                           mean_rmsds_arr - std_rmsds_arr,
                           mean_rmsds_arr + std_rmsds_arr,
                           alpha=0.3, color='#1f77b4', zorder=2)
            
            ax.set_xlabel('Number of Seeds', fontsize=12)
            ax.set_ylabel('Average Pose RMSD (Å)', fontsize=12)
            ax.set_title(f'{names[index]}' if index is not None else f'{pdb_id}', fontsize=16)
            ax.grid(True, alpha=0.3, zorder=1)
            ax.tick_params(labelsize=10)
            ax.text(0.65, 0.98, '(shaded: ±1 SD)', 
                    transform=ax.transAxes, fontsize=9, 
                    verticalalignment='top', alpha=0.7)
            # ymin, ymax = ax.get_ylim()
            # ax.set_ylim(ymin, ymax + 0.5)
        
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            axes[idx].text(0.5, 0.5, f'Error:\n{pdb_id}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
    
    # Hide unused subplots
    for idx in range(len(csv_files), 16):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'plots_out/pose_rmsd_bootstrap_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPose RMSD plot saved to: {output_file}")
    
    plt.close()

def plot_open_or_closed_rmsd(pdb_ids, names, af_pdb, rmsd_type='open'):
    """
    Create plot for open or closed RMSD comparing AF3 nolig vs AF3 lig
    
    Args:
        pdb_ids: List of PDB IDs to process
        names: List of names from pnas_table.csv
        af_pdb: List of PDB IDs from pnas_table.csv
        rmsd_type: 'open' or 'closed'
    """
    n_cases = min(len(pdb_ids), 16)
    n_rows = 4
    n_cols = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    axes = axes.flatten()
    
    # Process each case
    for idx, pdb_id in enumerate(pdb_ids[:16]):
        print(f"Processing {rmsd_type} RMSD for {pdb_id}...")
        
        index = af_pdb.index(pdb_id.upper()) if pdb_id.upper() in af_pdb else None
        
        ax = axes[idx]
        
        try:
            # Load data from both AF3 variants
            nolig_file = Path(f'pnas_af3_nolig/{pdb_id}/{rmsd_type}_rmsd.csv')
            lig_file = Path(f'pnas_af3_lig/{pdb_id}/{rmsd_type}_rmsd.csv')
            
            plotted_any = False
            
            # Process AF3 nolig
            if nolig_file.exists():
                seed_counts, mean_rmsds, std_rmsds = analyze_pdb_case_structs(nolig_file, n_bootstrap=100)
                
                seed_counts_arr = np.array(seed_counts)
                mean_rmsds_arr = np.array(mean_rmsds)
                std_rmsds_arr = np.array(std_rmsds)
                
                # Plot mean line
                ax.plot(seed_counts_arr, mean_rmsds_arr, 
                       marker='o', linestyle='-', markersize=4, linewidth=1.5,
                       color='#1f77b4', label='AF3 nolig', zorder=3)
                
                # Add shaded region for ±1 standard deviation
                ax.fill_between(seed_counts_arr, 
                               mean_rmsds_arr - std_rmsds_arr,
                               mean_rmsds_arr + std_rmsds_arr,
                               alpha=0.3, color='#1f77b4', zorder=2)
                plotted_any = True
            
            # Process AF3 lig
            if lig_file.exists():
                seed_counts, mean_rmsds, std_rmsds = analyze_pdb_case_structs(lig_file, n_bootstrap=100)

                seed_counts_arr = np.array(seed_counts)
                mean_rmsds_arr = np.array(mean_rmsds)
                std_rmsds_arr = np.array(std_rmsds)
                
                # Plot mean line
                ax.plot(seed_counts_arr, mean_rmsds_arr, 
                       marker='s', linestyle='-', markersize=4, linewidth=1.5,
                       color='#ff7f0e', label='AF3 lig', zorder=3)
                
                # Add shaded region for ±1 standard deviation
                ax.fill_between(seed_counts_arr, 
                               mean_rmsds_arr - std_rmsds_arr,
                               mean_rmsds_arr + std_rmsds_arr,
                               alpha=0.3, color='#ff7f0e', zorder=2)
                plotted_any = True
            
            if plotted_any:
                ax.set_xlabel('Number of Seeds', fontsize=12)
                ax.set_ylabel(f'Average  RMSD to {rmsd_type.capitalize()} Structure (Å)', fontsize=12)
                ax.set_title(f'{names[index]}' if index is not None else f'{pdb_id}', fontsize=16)
                ax.grid(True, alpha=0.3, zorder=1)
                ax.tick_params(labelsize=10)
                ax.legend(fontsize=10, loc='best')
                # Add text annotation for shaded region
                ax.text(0.05, 0.98, '(shaded: ±1 SD)', 
                       transform=ax.transAxes, fontsize=9, 
                       verticalalignment='top', alpha=0.7)
                ymin, ymax = ax.get_ylim()
                number = (ymax - ymin) / 10
                ax.set_ylim(ymin, ymax + number)
            else:
                ax.text(0.5, 0.5, f'No data:\n{pdb_id}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            axes[idx].text(0.5, 0.5, f'Error:\n{pdb_id}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
    
    # Hide unused subplots
    for idx in range(len(pdb_ids), 16):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'plots_out/{rmsd_type}_rmsd_bootstrap_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n{rmsd_type.capitalize()} RMSD plot saved to: {output_file}")
    
    plt.close()

def main():
    # Create output directory if it doesn't exist
    Path('plots_out').mkdir(exist_ok=True)
    
    # Load PDB names from pnas_table.csv
    df = pd.read_csv('pnas_table.csv')
    names = df['name'].tolist()
    af_pdb = [x.split('.')[0] for x in df['af_pdb'].tolist()]
    
    # ===== PLOT 1: Pose RMSD (original data) =====
    print("\n=== Processing Pose RMSD ===")
    data_dir = Path('pose_rmsds')
    if data_dir.exists():
        csv_files = sorted(data_dir.glob('*_lig_RMSDs.csv'))
        if len(csv_files) > 0:
            plot_pose_rmsd(csv_files, names, af_pdb)
        else:
            print(f"No CSV files found in {data_dir}")
    else:
        print(f"Warning: Directory '{data_dir}' not found!")
    
    # ===== Get list of PDB IDs from AF3 directories =====
    # Check pnas_af3_nolig directory for available PDB IDs
    nolig_dir = Path('pnas_af3_nolig')
    pdb_ids = []
    
    if nolig_dir.exists():
        pdb_dirs = sorted([d for d in nolig_dir.iterdir() if d.is_dir()])
        pdb_ids = [d.name for d in pdb_dirs]
    
    # Also check pnas_af3_lig directory and merge
    lig_dir = Path('pnas_af3_lig')
    if lig_dir.exists():
        lig_pdb_dirs = sorted([d for d in lig_dir.iterdir() if d.is_dir()])
        lig_pdb_ids = [d.name for d in lig_pdb_dirs]
        # Merge unique PDB IDs
        pdb_ids = sorted(list(set(pdb_ids + lig_pdb_ids)))
    
    if len(pdb_ids) == 0:
        print("Warning: No PDB directories found in pnas_af3_nolig or pnas_af3_lig")
        return
    
    print(f"\nFound {len(pdb_ids)} PDB cases in AF3 directories")
    
    # ===== PLOT 2: Open RMSD =====
    print("\n=== Processing Open RMSD ===")
    plot_open_or_closed_rmsd(pdb_ids, names, af_pdb, rmsd_type='open')
    
    # ===== PLOT 3: Closed RMSD =====
    print("\n=== Processing Closed RMSD ===")
    plot_open_or_closed_rmsd(pdb_ids, names, af_pdb, rmsd_type='closed')
    
    print("\n=== All plots completed! ===")

if __name__ == '__main__':
    main()