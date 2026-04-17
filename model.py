from xml.parsers.expat import model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pydicom
from tqdm import tqdm
from scipy.ndimage import zoom


class SimpleCNN(nn.Module):
    """
    Simple 3D CNN for medical image classification.
    """
    def __init__(self, num_classes=4, input_channels=1, input_shape=(64, 64, 64)):
        """
        Args:
            num_classes (int): Number of output classes (default: 4 for grades 1-4)
            input_channels (int): Number of input channels (default: 1 for grayscale)
            input_shape (tuple): Input shape (depth, height, width) - default (64, 64, 64)
        """
        super(SimpleCNN, self).__init__()
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=8, stride=4)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=4, stride=4)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.InstanceNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Calculate flattened size after 3 pooling layers (each divides by 2)
        # After 3 pooling layers, spatial dims are divided by 8 (2^3)
        depth_final = input_shape[0] // 8
        height_final = input_shape[1] // 8
        width_final = input_shape[2] // 8

        fc_input_size = 3200 # 128 * depth_final * height_final * width_final
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)
        """
        # Conv block 1
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Conv block 2
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Conv block 3
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def print_memory_usage(self):
        """
        Prints the estimated memory usage of the model parameters and buffers.
        """
        total_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        total_bytes += sum(b.numel() * b.element_size() for b in self.buffers())

        return total_bytes
    
    def train_model(self, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001, weight_decay=1e-4):

        criterion = nn.CrossEntropyLoss()
        # Added weight_decay for L2 Regularization
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels, patient_ids in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Train', unit='batch'):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = self(images)
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
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels, patient_ids in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} Val', unit='batch'):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = self(images)
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
