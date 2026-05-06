import torch
from torch.utils.data import Dataset
import sys

class DCMDataset(Dataset):
    """
    PyTorch Dataset for loading patient 3D scan series volumes and their labels.
    Supports both on-the-fly loading and preloaded data.
    """
    def __init__(self, series_data, target_shape, preloaded_data, transform=None):
        """
        Args:
            series_data (list): List of tuples (grade, [dcm_file_paths], patient_id, series_id)
            target_shape (tuple): Target shape for resizing 3D volumes (depth, height, width)
            transform (callable): Optional transforms to apply to the volume
            preloaded_data (dict): Optional preloaded data where keys are indices and values are (volume, label, patient_id)
        """
        self.series_data = series_data
        self.target_shape = target_shape
        self.transform = transform
        self.preloaded_data = preloaded_data
        
        for vol, _, _ in self.preloaded_data.values():
            if vol.dtype != torch.uint8:
                raise TypeError(f"Preloaded data must be of type torch.uint8.")

    def print_memory_usage(self):
        """
        Prints the estimated memory usage of the preloaded dataset.
        """
        total_bytes = 0

        for volume, label, patient_id in self.preloaded_data.values():
            total_bytes += volume.element_size() * volume.nelement()
            total_bytes += label.element_size() * label.nelement()
            total_bytes += sys.getsizeof(patient_id)
            

        return total_bytes

    def __len__(self):
        return len(self.series_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (volume, label, patient_id) where volume is a 3D tensor and label is the grade
        """
        volume, label, patient_id = self.preloaded_data[idx]
        
        # Convert back to float and normalize to 0-1 for the model on-the-fly
        volume = volume.float() * 0.00392156862745
            
        if self.transform:
            volume = self.transform(volume)
        return volume, label, patient_id
        