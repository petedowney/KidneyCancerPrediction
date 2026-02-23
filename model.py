import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pydicom
from tqdm import tqdm


class SimpleCNN(nn.Module):
    """
    Simple 3D CNN for medical image classification.
    """
    def __init__(self, num_classes=4, input_channels=1):
        """
        Args:
            num_classes (int): Number of output classes (default: 4 for grades 1-4)
            input_channels (int): Number of input channels (default: 1 for grayscale)
        """
        super(SimpleCNN, self).__init__()
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # For 64x64x64 input: after 3 pooling layers -> 8x8x8 spatial dims
        # For 128x128x128 input: after 3 pooling layers -> 16x16x16 spatial dims
        self.fc1 = nn.Linear(128 * 8 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)
        """
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class DCMDataset(Dataset):
    """
    PyTorch Dataset for loading patient 3D scan series volumes and their labels.
    Supports both on-the-fly loading and preloaded data.
    """
    def __init__(self, series_data, target_shape=(64, 64, 64), transform=None, preloaded_data=None):
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
        self.preloaded_data = preloaded_data or {}
        self.is_preloaded = len(self.preloaded_data) > 0
    
    def __len__(self):
        return len(self.series_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (volume, label, patient_id) where volume is a 3D tensor and label is the grade
        """
        # Return preloaded data if available
        if self.is_preloaded and idx in self.preloaded_data:
            volume, label, patient_id = self.preloaded_data[idx]
            if self.transform:
                volume = self.transform(volume)
            return volume, label, patient_id
        
        # Otherwise load from disk
        grade, dcm_file_paths, patient_id, series_id = self.series_data[idx]
        
        try:
            # Load all DICOM slices for this series
            slices = []
            for dcm_path in dcm_file_paths:
                try:
                    # Use force=True to read non-standard DICOM files
                    dicom_file = pydicom.dcmread(dcm_path, force=True)
                    if hasattr(dicom_file, 'pixel_array'):
                        image = dicom_file.pixel_array.astype(np.float32)
                        slices.append(image)
                except Exception as e:
                    # Silently skip problematic files
                    continue
            
            if not slices:
                raise ValueError(f"No valid DICOM slices found for series {series_id}")
            
            # Filter slices to ensure they all have the same shape
            slice_shape = slices[0].shape
            valid_slices = []
            for slice_data in slices:
                if slice_data.shape == slice_shape:
                    valid_slices.append(slice_data)
            
            if not valid_slices:
                raise ValueError(f"No slices with matching shape found for series {series_id}")
            
            # Stack slices into 3D volume
            volume = np.stack(valid_slices, axis=0)
            
            # Normalize volume
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
            
            # Resize to target shape
            from scipy.ndimage import zoom
            current_shape = volume.shape
            zoom_factors = [self.target_shape[i] / current_shape[i] for i in range(len(current_shape))]
            volume = zoom(volume, zoom_factors, order=1)
            volume = volume[:self.target_shape[0], :self.target_shape[1], :self.target_shape[2]]
            
            # Add channel dimension
            volume = np.expand_dims(volume, axis=0)
            
            # Convert to tensor
            volume = torch.from_numpy(volume).float()
            
            # Apply transforms if provided
            if self.transform:
                volume = self.transform(volume)
            
            # Convert grade to tensor (subtract 1 if grades are 1-4, to make them 0-3)
            label = torch.tensor(grade - 1, dtype=torch.long)
            
            return volume, label, patient_id
        
        except Exception as e:
            # Return a zero tensor as fallback (silently skip series with no usable data)
            zero_volume = torch.zeros((1, *self.target_shape), dtype=torch.float32)
            label = torch.tensor(0, dtype=torch.long)
            return zero_volume, label, patient_id


def preload_dcm_files(dcm_data, target_shape=(64, 64, 64), max_files=None):
    """
    Preload all DICOM patient scan series into memory as 3D volumes.
    Each series from each patient is stacked into a 3D volume.
    
    Args:
        dcm_data (list): List of tuples (grade, [dcm_file_paths], patient_id, series_id)
        target_shape (tuple): Target shape for resizing volumes (default: 64x64x64 for speed)
        max_files (int): Maximum number of series to load (for testing, set to subset)
        
    Returns:
        dict: Dictionary mapping index to (volume, label, patient_id) tuples
    """
    # Limit series if requested
    if max_files is not None:
        data_to_load = dcm_data[:max_files]
        print(f"Loading {len(data_to_load)} scan series (limited by max_files={max_files})")
    else:
        data_to_load = dcm_data
        print(f"Loading all {len(data_to_load)} scan series")
    
    preloaded = {}
    
    for idx in tqdm(range(len(data_to_load)), desc='Preloading Scan Series', unit='series'):
        grade, dcm_file_paths, patient_id, series_id = data_to_load[idx]
        
        try:
            # Load all DICOM slices for this series
            slices = []
            valid_slices = []
            
            for dcm_path in dcm_file_paths:
                try:
                    # Use force=True to read non-standard DICOM files
                    dicom_file = pydicom.dcmread(dcm_path, force=True)
                    if hasattr(dicom_file, 'pixel_array'):
                        image = dicom_file.pixel_array.astype(np.float32)
                        slices.append(image)
                except Exception as e:
                    # Silently skip problematic files
                    continue
            
            if not slices:
                raise ValueError(f"No valid DICOM slices found for patient {patient_id} series {series_id}")
            
            # Filter slices to ensure they all have the same shape
            # (some series have scout images or other non-standard slices)
            slice_shape = slices[0].shape  # 2D shape of individual DICOM slices
            for slice_data in slices:
                if slice_data.shape == slice_shape:
                    valid_slices.append(slice_data)
            
            if not valid_slices:
                raise ValueError(f"No slices with matching shape found for series {series_id}")
            
            # Stack slices into 3D volume (depth, height, width)
            volume = np.stack(valid_slices, axis=0)
            
            # Normalize volume
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
            
            # Resize to target shape using interpolation
            from scipy.ndimage import zoom
            current_shape = volume.shape
            zoom_factors = [target_shape[i] / current_shape[i] for i in range(len(current_shape))]
            volume = zoom(volume, zoom_factors, order=1)
            volume = volume[:target_shape[0], :target_shape[1], :target_shape[2]]
            
            # Add channel dimension (1, depth, height, width)
            volume = np.expand_dims(volume, axis=0)
            
            # Convert to tensor
            volume_tensor = torch.from_numpy(volume).float()
            label = torch.tensor(grade - 1, dtype=torch.long)
            
            preloaded[idx] = (volume_tensor, label, patient_id)
        
        except Exception as e:
            # Store zero tensor as fallback (silently skip series with no usable data)
            zero_volume = torch.zeros((1, *target_shape), dtype=torch.float32)
            label = torch.tensor(0, dtype=torch.long)
            preloaded[idx] = (zero_volume, label, patient_id)
    
    return preloaded


def create_data_loader(series_data, batch_size=4, shuffle=True, num_workers=0, target_shape=(64, 64, 64), preloaded_data=None):
    """
    Create a DataLoader for the patient scan series 3D volume dataset.
    
    Args:
        series_data (list): List of tuples (grade, [dcm_file_paths], patient_id, series_id)
        batch_size (int): Batch size for the loader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for loading data
        target_shape (tuple): Target shape for resizing volumes
        preloaded_data (dict): Optional preloaded data dictionary
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = DCMDataset(series_data, target_shape=target_shape, preloaded_data=preloaded_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader


def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001):
    """
    Train and validate a model.
    
    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        device (torch.device): Device to train on (cuda or cpu)
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate for the optimizer
        
    Returns:
        dict: Dictionary containing training history with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels, patient_ids in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Train', unit='batch'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, patient_ids in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} Val', unit='batch'):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        
        # Store history
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%")
    
    print("Training complete!")
    return history
