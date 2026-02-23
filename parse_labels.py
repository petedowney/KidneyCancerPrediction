"""
Parse labels.txt and create a hash table mapping bcr_patient_barcode to tumor_grade.
"""

import os


def find_dcm_files(grade, root_path):
    """
    Recursively find all .dcm (DICOM) files in a directory and create tuples with grade.
    
    Args:
        grade: The cancer grade to associate with the files
        root_path (str): Root directory path to search for .dcm files
        
    Returns:
        list: List of tuples (grade, dcm_file_path) for all .dcm files found
    """
    dcm_tuples = []
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith('.dcm'):
                full_path = os.path.join(dirpath, filename)
                dcm_tuples.append((grade, full_path))
    
    return dcm_tuples


def parse_labels(filepath):
    """
    Parse the labels.txt file and create a dictionary mapping bcr_patient_barcode to tumor_grade.
    
    Args:
        filepath (str): Path to the labels.txt file
        
    Returns:
        dict: Dictionary with bcr_patient_barcode as key and tumor_grade as value
    """
    grade_map = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header rows (lines 0-2: full header, alternative names, and CDE_ID)
    # Data starts at line 3
    for line in lines[3:]:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        
        fields = line.split('\t')
        
        # Column index 1 is bcr_patient_barcode
        # Column index 4 is tumor_grade
        if len(fields) > 4:
            barcode = fields[1]
            grade = fields[4]
            grade_map[barcode] = int(grade[1:])
    
    return grade_map


def get_all_dcm_files(labels_filepath, data_root_path):
    """
    Parse labels and group all DCM files by patient AND scan series.
    Each unique LEAF folder becomes one 3D volume sample.
    
    Series are determined by folder structure - only leaf folders (with DICOM files
    but no subdirectories containing DICOM files) are treated as separate series.
    
    Args:
        labels_filepath (str): Path to the labels.txt file
        data_root_path (str): Root path to the TCGA data directory containing patient folders
        
    Returns:
        list: List of tuples (grade, [dcm_file_paths], patient_id, series_id)
              Each tuple represents one scan series from one patient
    """
    grade_map = parse_labels(labels_filepath)
    series_data = []
    
    # Iterate through each barcode and its corresponding grade
    for barcode, grade in grade_map.items():
        # Construct the patient directory path
        patient_dir = os.path.join(data_root_path, barcode)
        
        # Check if the patient directory exists
        if os.path.isdir(patient_dir):
            # Find all folders with DICOM files
            all_dcm_folders = {}
            
            for dirpath, dirnames, filenames in os.walk(patient_dir):
                # Collect DICOM files in this folder
                dcm_files = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith('.dcm')]
                
                if dcm_files:
                    all_dcm_folders[dirpath] = sorted(dcm_files)
            
            # Filter to only LEAF folders (folders that don't have subdirectories with DICOM files)
            leaf_folders = {}
            for folder_path, dcm_paths in all_dcm_folders.items():
                # Check if any other folder is a subdirectory of this one
                is_leaf = True
                for other_folder in all_dcm_folders.keys():
                    if other_folder.startswith(folder_path + os.sep) and other_folder != folder_path:
                        # This folder has a subdirectory with DICOM files, so it's not a leaf
                        is_leaf = False
                        break
                
                if is_leaf:
                    leaf_folders[folder_path] = dcm_paths
            
            # Create one sample per series (leaf folder)
            for series_id, dcm_paths in leaf_folders.items():
                if dcm_paths:
                    series_data.append((grade, dcm_paths, barcode, series_id))
    
    return series_data



