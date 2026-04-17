import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pydicom
from tqdm import tqdm
from scipy.ndimage import zoom
from dataset import DCMDataset

def get_orientation(orientation_values):
    if not orientation_values or len(orientation_values) != 6:
        return "Unknown"
    x_vector = np.array(orientation_values[:3])
    y_vector = np.array(orientation_values[3:])
    z_vector = np.cross(x_vector, y_vector)
    abs_z = np.abs(z_vector)
    max_idx = np.argmax(abs_z)
    if max_idx == 0:
        return "Sagittal (Vertical)"
    elif max_idx == 1:
        return "Coronal (Vertical)"
    else:
        return "Axial (Horizontal)"

def preload_dcm_files(dcm_data, target_shape=(64, 64, 64), max_files=None, crop=None):
    class_items = [[] for _ in range(4)]
    for item in dcm_data:
        grade = item[0]
        class_items[grade - 1].append(item)
        
    if max_files is None:
        max_files = len(dcm_data)
    else:
        max_files = min(max_files, len(dcm_data))
        
    preloaded = {}
    allocated = [0, 0, 0, 0]
    class_indices = [0, 0, 0, 0]
    total_loaded = 0
    
    pbar = tqdm(total=max_files, desc='Preloading Scan Series', unit='series')
    
    while total_loaded < max_files:
        available_classes = [i for i in range(4) if class_indices[i] < len(class_items[i])]
        if not available_classes:
            break
            
        # Sort available classes by allocation to keep them balanced
        available_classes.sort(key=lambda i: allocated[i])
        cls = available_classes[0]
        
        item = class_items[cls][class_indices[cls]]
        class_indices[cls] += 1
        
        grade, dcm_file_paths, patient_id, series_id = item
        
        slices = []
        skip_series = False
        orientation_found = False
        for dcm_path in dcm_file_paths:
            dicom_file = pydicom.dcmread(dcm_path, force=True)
            
            if not orientation_found and hasattr(dicom_file, 'ImageOrientationPatient'):
                orientation = get_orientation(dicom_file.ImageOrientationPatient)
                orientation_found = True
                if orientation != "Axial (Horizontal)":
                    skip_series = True
                    break
                
            if hasattr(dicom_file, 'pixel_array'):
                slices.append(dicom_file.pixel_array.astype(np.float32))

        if skip_series or not orientation_found:
            continue

        if not slices:
            continue

        slice_shape = slices[0].shape
        valid_slices = [s for s in slices if s.shape == slice_shape]

        if not valid_slices:
            continue

        volume = np.stack(valid_slices, axis=0)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        if crop:
            crop_z, crop_x, crop_y = crop

            # Support both fraction (e.g. 0.1) and percentage (e.g. 10) formats
            if crop_x > 1: crop_x /= 100.0
            if crop_y > 1: crop_y /= 100.0
            if crop_z > 1: crop_z /= 100.0
            
            d, h, w = volume.shape
            drop_x = int(w * crop_x)
            drop_y = int(h * crop_y)
            drop_z = int(d * crop_z)
            
            # Crop symmetrically from edges
            start_x = drop_x // 2
            end_x = w - (drop_x - start_x)
            start_y = drop_y // 2
            end_y = h - (drop_y - start_y)
            start_z = drop_z // 2
            end_z = d - (drop_z - start_z)
            
            volume = volume[start_z:end_z, start_y:end_y, start_x:end_x]

        # Subsample Z-axis (depth) without interpolating by selecting nearest slices directly
        current_shape = volume.shape
        z_indices = np.round(np.linspace(0, current_shape[0] - 1, target_shape[0])).astype(int)
        volume = volume[z_indices, :, :]

        # Now only zoom/interpolate the X and Y axes
        zoom_factors = [1.0, target_shape[1] / current_shape[1], target_shape[2] / current_shape[2]]
        volume = zoom(volume, zoom_factors, order=1)
        volume = volume[:target_shape[0], :target_shape[1], :target_shape[2]]

        volume = np.expand_dims(volume, axis=0)
        volume_tensor = torch.from_numpy(volume).float()
        label = torch.tensor(grade - 1, dtype=torch.long)

        preloaded[total_loaded] = (volume_tensor, label, patient_id)
        
        allocated[cls] += 1
        total_loaded += 1
        pbar.update(1)

    pbar.close()
    return preloaded


def create_data_loader(series_data, batch_size=4, shuffle=True, num_workers=0, target_shape=(64, 64, 64), preloaded_data=None):
    """
    Create a DataLoader for the patient scan series 3D volume dataset.
    """
    dataset = DCMDataset(series_data, target_shape=target_shape, preloaded_data=preloaded_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader
