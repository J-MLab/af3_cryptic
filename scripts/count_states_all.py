import os
import pandas as pd

"""
-Script to count the open and closed conformations of the AF3 generated and PDB structures based on rmsd --> NEW VERSION
-better than the other two count_rmsd
-Run from af3_testing directory
Author: Felix T.
"""


# helper function for the helper function to find the rmsd of something given the name
def scan_CSV(csv_path, PDB_ID):
    """
    Parameters:
    csv_path: path to closed_rmsd.csv file
    PDB_ID: PDB ID you want to extract the rmsd of from the CSV
    """
    try:
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_path)

        # Filter the DataFrame for rows where the 'model' column matches the given model
        matching_rows = df[
            df["model"].astype(str).str.strip().str[:4].str.upper()
            == PDB_ID[:4].strip().upper()
        ]

        if matching_rows.empty:
            print(
                f"No entry found for model starting with '{PDB_ID[:4].strip().upper()}'."
            )
            return None

        # Return the RMSD value from the first matching row.
        # Adjust the column name 'rmsd' if needed based on your CSV's header.
        rmsd_value = matching_rows.iloc[0]["loop_rmsd"]
        return rmsd_value
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# helper function to find the rmsd of a struct based off bound/unbound
def find_rmsd(PDB_ID):
    """
    Parameters:
    PDB_ID: 4 digit PDB code in the af_pdb column
    """

    PDB_path = f"pdb_structures_final/{PDB_ID}_pdbs"

    # first find the open PDB ID for passed in AF3_PDB
    df = pd.read_csv("pnas_table_mod.csv")
    af_pdb_value = PDB_ID
    matching_rows = df[df["af_pdb"].str[:4] == PDB_ID]
    if not matching_rows.empty:
        # Get the first matching value from the 'open' column.
        open_pdb = matching_rows.iloc[0]["open"]
        print(f"open_pdb for af_pdb '{af_pdb_value}' is: {open_pdb}")
    else:
        print(f"No matching entry found for af_pdb '{af_pdb_value}'.")

    # find the RMSD of the open_pdb
    if  PDB_ID != "3IXJ":  # the two cases where it won't be bound
        csv_path = os.path.join(PDB_path, "bound/closed_rmsd.csv")

        return scan_CSV(csv_path, open_pdb)
    else:
        csv_path = os.path.join(PDB_path, "unbound/closed_rmsd.csv")

        return scan_CSV(csv_path, open_pdb)


# helper function for the helper function below
def state_counter(path_to_csv, state_counts, key, threshold):
    """
    Reads the CSV file and counts the number of rows where the 'loop_rmsd'
    value is less than or equal to the threshold.

    Args:
        path_to_csv (str): Path to the CSV file.
        state_counts (dict): Dictionary to update with the count.
        key (str): The key under which the count is stored.
        threshold (float): The threshold value for a conformation.

    Returns:
        int: Total number of RMSD rows read from the CSV.
    """
    print(f"Analyzing {key} in file {path_to_csv} using threshold {threshold}")
    df = pd.read_csv(path_to_csv)
    # Count how many rows have loop_rmsd less than or equal to threshold
    state_counts[key] = len(df[df["loop_rmsd"] <= threshold])
    # Return total count of RMSD values (used for calculating 'neither')
    return df["loop_rmsd"].count()


# helper function to count the conformation states for the pdb folders
def update_state_counts_from_folder(state_counts, folder_path, pdb_id, threshold):
    """
    Updates the state_counts dictionary with the counts of closed, open, and neither
    conformations from the given folder. This folder must contain subfolders 'bound'
    and 'unbound'. Each subfolder is expected to contain two CSV files:
    'closed_rmsd.csv' and 'open_rmsd.csv'.

    The dictionary keys are updated in the following format:
        - "pdb_bound_<pdb_id>_closed"
        - "pdb_bound_<pdb_id>_open"
        - "pdb_bound_<pdb_id>_neither"
        - "pdb_unbound_<pdb_id>_closed"
        - "pdb_unbound_<pdb_id>_open"
        - "pdb_unbound_<pdb_id>_neither"

    Args:
        state_counts (dict): Dictionary that will be updated with the counts.
        folder_path (str): The path to the folder that contains the 'bound'
                           and 'unbound' subdirectories.
        pdb_id (str): The pdb identifier.
        threshold (float): The threshold used to determine if a conformation is open or closed.

    Returns:
        dict: The updated state_counts dictionary.
    """
    # Process both 'bound' and 'unbound' subfolders
    for state in ["bound", "unbound"]:
        state_dir = os.path.join(folder_path, state)
        if not os.path.isdir(state_dir):
            print(f"Directory {state_dir} does not exist. Skipping state '{state}'.")
            continue

        # Initialize variables to store paths for CSV files.
        closed_csv = None
        open_csv = None

        # Look for the required CSV files in the current state directory.
        for filename in os.listdir(state_dir):
            if filename == "closed_rmsd.csv":
                closed_csv = os.path.join(state_dir, filename)
            elif filename == "open_rmsd.csv":
                open_csv = os.path.join(state_dir, filename)

        # If any of the CSV files are missing, skip this state.
        if not closed_csv or not open_csv:
            print(
                f"Missing CSV file(s) in {state_dir} for pdb {pdb_id}. Skipping state '{state}'."
            )
            continue

        # Define the dictionary keys for the current state and pdb_id.
        key_closed = f"pdb_{state}_{pdb_id}_closed"
        key_open = f"pdb_{state}_{pdb_id}_open"
        key_neither = f"pdb_{state}_{pdb_id}_neither"

        # Count the 'closed' conformations and get the total RMSD count from the closed CSV.
        total_rmsds = state_counter(closed_csv, state_counts, key_closed, threshold)
        # Count the 'open' conformations from the open CSV.
        state_counter(open_csv, state_counts, key_open, threshold)
        # Any remaining RMSDs (from closed CSV total) are classified as 'neither'.
        state_counts[key_neither] = total_rmsds - (
            state_counts[key_closed] + state_counts[key_open]
        )

    return state_counts


# helper function to count the state counts from af3
def update_af3_state_counts(state_counts, folder_path, pdb_id, threshold):
    """
    Updates the state_counts dictionary with the counts of closed, open, and neither
    conformations, based on the provided RMSD threshold.

    The expected folder structure is:

        folder_path/
            └── {pdb_id}_pdbs/
                  ├── closed_rmsd.csv
                  └── open_rmsd.csv

    The folder_path provided will be one of the two directories: one for lig (bound) or
    one for nolig (unbound). Based on that, the keys in state_counts will be:

        For lig:  af3_bound_{pdb_id}_closed, af3_bound_{pdb_id}_open, af3_bound_{pdb_id}_neither
        For nolig: af3_unbound_{pdb_id}_closed, af3_unbound_{pdb_id}_open, af3_unbound_{pdb_id}_neither

    Args:
        state_counts (dict): Dictionary to update.
        folder_path (str): Path to the directory (e.g., "/projectnb/protmd/felix/af3_testing/pnas_af3_lig"
                           or "/projectnb/protmd/felix/af3_testing/pnas_af3_nolig").
        pdb_id (str): The 4-letter PDB identifier.
        threshold (float): The threshold used to classify a conformation as open or closed.

    Returns:
        dict: The updated state_counts dictionary.
    """
    # Determine if folder_path indicates bound (lig) or unbound (nolig)
    if "_lig" in folder_path:
        state_str = "bound"
    elif "_nolig" in folder_path:
        state_str = "unbound"
    else:
        print("Folder path does not indicate 'lig' or 'nolig'.")
        state_str = "unknown"

    # Construct the PDB folder path
    pdb_folder = folder_path  # leaving like this so I don't have to change many other variable names
    if not os.path.isdir(pdb_folder):
        print(f"PDB folder not found: {pdb_folder}")
        return state_counts

    # Initialize variables for the CSV file paths
    closed_csv = None
    open_csv = None

    # Iterate over files in the pdb_folder
    for filename in os.listdir(pdb_folder):
        if filename == "closed_rmsd.csv":
            closed_csv = os.path.join(pdb_folder, filename)
        elif filename == "open_rmsd.csv":
            open_csv = os.path.join(pdb_folder, filename)

    # Check that both CSV files exist
    if not closed_csv or not open_csv:
        print(f"Missing CSV file(s) in {pdb_folder}. Skipping pdb {pdb_id}.")
        return state_counts

    # Format the keys for state_counts according to the requested pattern.
    key_closed = f"af3_{state_str}_{pdb_id}_closed"
    key_open = f"af3_{state_str}_{pdb_id}_open"
    key_neither = f"af3_{state_str}_{pdb_id}_neither"

    # Count the closed conformations and get the total RMSD count from the closed CSV.
    total_rmsds = state_counter(closed_csv, state_counts, key_closed, threshold)
    # Count the open conformations from the open CSV.
    state_counter(open_csv, state_counts, key_open, threshold)
    # Those that are neither open nor closed.
    state_counts[key_neither] = total_rmsds - (
        state_counts[key_closed] + state_counts[key_open]
    )

    return state_counts


# actual code to get results
pnas_table = pd.read_csv("pnas_table_mod.csv")
state_counts = {}

for pdb in pnas_table["af_pdb"]:
    pdb = pdb[:4]
    rmsd = find_rmsd(pdb)

    state_counts[f"rmsd/2_{pdb}"] = rmsd / 2

    # call function to update dictionary with PDB values
    update_state_counts_from_folder(
        state_counts, f"pdb_structures_final/{pdb}_pdbs", pdb, float(rmsd) / 2
    )

    # call function to update dictionary with AF3 values of lig and nolig
    update_af3_state_counts(
        state_counts, f"pnas_af3_lig/{pdb.lower()}", pdb, float(rmsd) / 2
    )
    update_af3_state_counts(
        state_counts, f"pnas_af3_nolig/{pdb.lower()}", pdb, float(rmsd) / 2
    )


# save the dictionary to a csv and output it
output = pd.DataFrame(list(state_counts.items()), columns=["State", "Count"])

csv_file_path = "total_state_counts_mod.csv"
output.to_csv(csv_file_path, index=False)

print(f"CSV file created successfully at: {csv_file_path}")
