import torch
import copy
import sys
import gc
import os
import torch
import traceback
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import logging
import os
import json
import zipfile
import tarfile
import gzip
import bz2
import lzma
from datetime import datetime, timedelta
import time
import shutil
import glob
from tqdm import tqdm
import random
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from pathlib import Path
import torch.multiprocessing
from abc import ABC, abstractmethod

# Set sharing strategy at the start
torch.multiprocessing.set_sharing_strategy('file_system')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction models"""
    def __init__(self, config: Dict, device: str = None):
        """Initialize base feature extractor"""
        self.config = self.verify_config(config)

        # Set device
        if device is None:
            self.device = torch.device('cuda' if self.config['execution_flags']['use_gpu']
                                     and torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize common parameters
        self.feature_dims = self.config['model']['feature_dims']
        self.learning_rate = self.config['model'].get('learning_rate', 0.001)

        # Initialize training metrics
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.history = defaultdict(list)
        self.training_log = []
        self.training_start_time = time.time()

        # Setup logging directory
        self.log_dir = os.path.join('Traininglog', self.config['dataset']['name'])
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize model
        self.feature_extractor = self._create_model()

        # Load checkpoint or initialize optimizer
        if not self.config['execution_flags'].get('fresh_start', False):
            self._load_from_checkpoint()

        # Initialize optimizer if not created during checkpoint loading
        if not hasattr(self, 'optimizer'):
            self.optimizer = self._initialize_optimizer()
            logger.info(f"Initialized {self.optimizer.__class__.__name__} optimizer")

        # Initialize scheduler
        self.scheduler = None
        if self.config['model'].get('scheduler'):
            self.scheduler = self._initialize_scheduler()
            if self.scheduler:
                logger.info(f"Initialized {self.scheduler.__class__.__name__} scheduler")

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create and return the feature extraction model"""
        pass

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer based on configuration"""
        optimizer_config = self.config['model'].get('optimizer', {})

        # Set base parameters
        optimizer_params = {
            'lr': self.learning_rate,
            'weight_decay': optimizer_config.get('weight_decay', 1e-4)
        }

        # Configure optimizer-specific parameters
        optimizer_type = optimizer_config.get('type', 'Adam')
        if optimizer_type == 'SGD':
            optimizer_params['momentum'] = optimizer_config.get('momentum', 0.9)
            optimizer_params['nesterov'] = optimizer_config.get('nesterov', False)
        elif optimizer_type == 'Adam':
            optimizer_params['betas'] = (
                optimizer_config.get('beta1', 0.9),
                optimizer_config.get('beta2', 0.999)
            )
            optimizer_params['eps'] = optimizer_config.get('epsilon', 1e-8)

        # Get optimizer class
        try:
            optimizer_class = getattr(optim, optimizer_type)
        except AttributeError:
            logger.warning(f"Optimizer {optimizer_type} not found, using Adam")
            optimizer_class = optim.Adam
            optimizer_type = 'Adam'

        # Create and return optimizer
        optimizer = optimizer_class(
            self.feature_extractor.parameters(),
            **optimizer_params
        )

        logger.info(f"Initialized {optimizer_type} optimizer with parameters: {optimizer_params}")
        return optimizer

    def _initialize_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler if specified in config"""
        scheduler_config = self.config['model'].get('scheduler', {})
        if not scheduler_config:
            return None

        scheduler_type = scheduler_config.get('type')
        if not scheduler_type:
            return None

        try:
            if scheduler_type == 'StepLR':
                return optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 7),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_type == 'ReduceLROnPlateau':
                return optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=scheduler_config.get('factor', 0.1),
                    patience=scheduler_config.get('patience', 10),
                    verbose=True
                )
            elif scheduler_type == 'CosineAnnealingLR':
                return optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.get('T_max', self.config['training']['epochs']),
                    eta_min=scheduler_config.get('eta_min', 0)
                )
        except Exception as e:
            logger.warning(f"Failed to initialize scheduler: {str(e)}")
            return None

        return None

    def verify_config(self, config: Dict) -> Dict:
        """Verify and fill in missing configuration values"""
        if 'dataset' not in config:
            raise ValueError("Configuration must contain 'dataset' section")

        # Ensure all required sections exist
        required_sections = ['dataset', 'model', 'training', 'execution_flags']
        for section in required_sections:
            if section not in config:
                config[section] = {}

        # Verify model section
        model = config.setdefault('model', {})
        model.setdefault('feature_dims', 128)
        model.setdefault('learning_rate', 0.001)
        model.setdefault('encoder_type', 'cnn')

        # Verify training section
        training = config.setdefault('training', {})
        training.setdefault('batch_size', 32)
        training.setdefault('epochs', 20)
        training.setdefault('num_workers', min(4, os.cpu_count() or 1))
        training.setdefault('checkpoint_dir', os.path.join('Model', 'checkpoints'))

        # Verify execution flags
        exec_flags = config.setdefault('execution_flags', {})
        exec_flags.setdefault('mode', 'train_and_predict')
        exec_flags.setdefault('use_gpu', torch.cuda.is_available())
        exec_flags.setdefault('mixed_precision', True)
        exec_flags.setdefault('distributed_training', False)
        exec_flags.setdefault('debug_mode', False)
        exec_flags.setdefault('use_previous_model', True)
        exec_flags.setdefault('fresh_start', False)

        return config

    @abstractmethod
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch"""
        pass

    @abstractmethod
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        pass

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """Train the model"""
        early_stopping = self.config['training'].get('early_stopping', {})
        patience = early_stopping.get('patience', 5)
        min_delta = early_stopping.get('min_delta', 0.001)
        max_epochs = self.config['training']['epochs']

        patience_counter = 0
        best_val_metric = float('inf')

        try:
            for epoch in range(self.current_epoch, max_epochs):
                self.current_epoch = epoch

                # Training
                train_loss, train_acc = self._train_epoch(train_loader)

                # Validation
                val_loss, val_acc = None, None
                if val_loader:
                    val_loss, val_acc = self._validate(val_loader)
                    current_metric = val_loss
                else:
                    current_metric = train_loss

                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    else:
                        self.scheduler.step()

                # Log metrics
                self.log_training_metrics(epoch, train_loss, train_acc, val_loss, val_acc,
                                       train_loader, val_loader)

                # Save checkpoint
                self._save_checkpoint(is_best=False)

                # Check for improvement
                if current_metric < best_val_metric - min_delta:
                    best_val_metric = current_metric
                    patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

                # Memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return self.history

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def log_training_metrics(self, epoch: int, train_loss: float, train_acc: float,
                           test_loss: Optional[float] = None, test_acc: Optional[float] = None,
                           train_loader: Optional[DataLoader] = None,
                           test_loader: Optional[DataLoader] = None):
        """Log training metrics"""
        elapsed_time = time.time() - self.training_start_time

        metrics = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_time': elapsed_time,
            'elapsed_time_formatted': str(timedelta(seconds=int(elapsed_time))),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_samples': len(train_loader.dataset) if train_loader else None,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_samples': len(test_loader.dataset) if test_loader else None
        }

        self.training_log.append(metrics)

        log_df = pd.DataFrame(self.training_log)
        log_path = os.path.join(self.log_dir, 'training_metrics.csv')
        log_df.to_csv(log_path, index=False)

        logger.info(f"Epoch {epoch + 1}: "
                   f"Train Loss {train_loss:.4f}, Acc {train_acc:.2f}%" +
                   (f", Test Loss {test_loss:.4f}, Acc {test_acc:.2f}%"
                    if test_loss is not None else ""))

    @abstractmethod
    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from data loader"""
        pass

    def save_features(self, features: torch.Tensor, labels: torch.Tensor, output_path: str):
        """Save extracted features to CSV"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            chunk_size = 1000
            total_samples = features.shape[0]

            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)

                feature_dict = {
                    f'feature_{i}': features[start_idx:end_idx, i].numpy()
                    for i in range(features.shape[1])
                }
                feature_dict['target'] = labels[start_idx:end_idx].numpy()

                df = pd.DataFrame(feature_dict)
                mode = 'w' if start_idx == 0 else 'a'
                header = start_idx == 0

                df.to_csv(output_path, mode=mode, index=False, header=header)

                del feature_dict, df
                gc.collect()

            logger.info(f"Features saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise

class FeatureExtractorCNN(nn.Module):
    """CNN-based feature extractor model"""
    def __init__(self, in_channels: int = 3, feature_dims: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, feature_dims)
        self.batch_norm = nn.BatchNorm1d(feature_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.batch_norm(x)
        return x

class DynamicAutoencoder(nn.Module):
    """
    A flexible autoencoder that can handle n-dimensional inputs and produces
    flattened embeddings compatible with the CNN feature extractor output.
    """
    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int):
        super().__init__()

        self.input_shape = input_shape
        self.feature_dims = feature_dims
        self.n_dims = len(input_shape) - 1  # Subtract 1 for channels

        # Store progression of spatial dimensions
        self.spatial_dims = []
        current_dims = list(input_shape[1:])
        self.spatial_dims.append(current_dims.copy())

        # Calculate layer sizes and dimensions
        self.layer_sizes = []
        current_channels = input_shape[0]
        next_channels = 32  # Start with 32 channels

        # Calculate maximum possible downsampling layers
        min_dim = min(input_shape[1:])
        max_layers = int(np.log2(min_dim)) - 2
        max_layers = max(1, min(max_layers, 3))  # At least 1, at most 3 layers

        # Build encoder channel progression
        for _ in range(max_layers):
            self.layer_sizes.append(next_channels)
            # Update spatial dimensions for next layer
            current_dims = [(d + 2 - 3 + 1) // 2 for d in current_dims]  # Using stride 2
            self.spatial_dims.append(current_dims.copy())
            current_channels = next_channels
            next_channels = min(next_channels * 2, 256)

        logger.info(f"Channel progression: {[input_shape[0]] + self.layer_sizes}")
        logger.info(f"Spatial dimensions: {self.spatial_dims}")

        # Calculate final flattened size
        self.flattened_size = self.layer_sizes[-1] * np.prod(self.spatial_dims[-1])
        logger.info(f"Flattened size: {self.flattened_size}")

        # Build encoder
        self.encoder_layers = nn.ModuleList()
        current_channels = input_shape[0]

        for size in self.layer_sizes:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, size, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(size),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            current_channels = size

        # Build embedder
        self.embedder = nn.Sequential(
            nn.Linear(self.flattened_size, feature_dims),
            nn.BatchNorm1d(feature_dims),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Build unembedder
        self.unembedder = nn.Sequential(
            nn.Linear(feature_dims, self.flattened_size),
            nn.BatchNorm1d(self.flattened_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Build decoder
        self.decoder_layers = nn.ModuleList()
        reversed_sizes = [self.layer_sizes[0]] + [input_shape[0]]  # Add input channels at the end

        for i in range(len(self.layer_sizes) - 1, -1, -1):
            out_channels = reversed_sizes[len(self.layer_sizes) - 1 - i]
            in_channels = self.layer_sizes[i]

            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True) if i > 0 else nn.Tanh()
                )
            )

    def _calculate_flattened_size(self) -> int:
        """Calculate size of flattened feature maps before linear layer"""
        reduction_factor = 2 ** (len(self.layer_sizes) - 1)
        reduced_dims = [dim // reduction_factor for dim in self.spatial_dims]
        return self.layer_sizes[-1] * np.prod(reduced_dims)

    def _calculate_layer_sizes(self) -> List[int]:
        """Calculate progressive channel sizes for encoder/decoder"""
        # Start with input channels
        base_channels = 32  # Reduced from 64 to handle smaller images
        sizes = []
        current_size = base_channels

        # Calculate maximum number of downsampling layers based on smallest spatial dimension
        min_dim = min(self.input_shape[1:])
        max_layers = int(np.log2(min_dim)) - 2  # Ensure we don't reduce too much

        for _ in range(max_layers):
            sizes.append(current_size)
            if current_size < 256:  # Reduced from 512 to handle smaller images
                current_size *= 2

        logger.info(f"Layer sizes: {sizes}")
        return sizes


    def _create_conv_block(self, in_channels: int, out_channels: int, **kwargs) -> nn.Sequential:
        """Create a convolutional block with batch norm and activation"""
        conv_class = nn.Conv1d if self.n_dims == 1 else (
            nn.Conv2d if self.n_dims == 2 else nn.Conv3d
        )

        return nn.Sequential(
            conv_class(in_channels, out_channels, **kwargs),
            nn.BatchNorm1d(out_channels) if self.n_dims == 1 else (
                nn.BatchNorm2d(out_channels) if self.n_dims == 2 else
                nn.BatchNorm3d(out_channels)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _create_deconv_block(self, in_channels: int, out_channels: int, **kwargs) -> nn.Sequential:
        """Create a deconvolutional block with batch norm and activation"""
        deconv_class = nn.ConvTranspose1d if self.n_dims == 1 else (
            nn.ConvTranspose2d if self.n_dims == 2 else nn.ConvTranspose3d
        )

        return nn.Sequential(
            deconv_class(in_channels, out_channels, **kwargs),
            nn.BatchNorm1d(out_channels) if self.n_dims == 1 else (
                nn.BatchNorm2d(out_channels) if self.n_dims == 2 else
                nn.BatchNorm3d(out_channels)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embedding space"""
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Flatten and embed
        x = x.view(x.size(0), -1)
        return self.embedder(x)


    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode embedding back to input space"""
        # Unembed and reshape
        x = self.unembedder(x)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.spatial_dims[-1][0], self.spatial_dims[-1][1])

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x)

        return x

    def _get_spatial_shape(self) -> Tuple[int, ...]:
        """Get spatial dimensions after encoding"""
        reduction_factor = 2 ** (len(self.layer_sizes) - 1)
        return tuple(dim // reduction_factor for dim in self.spatial_dims)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both embedding and reconstruction"""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return embedding, reconstruction

    def get_encoding_shape(self) -> Tuple[int, ...]:
        """Get the shape of the encoding at each layer"""
        return tuple([size for size in self.layer_sizes])

    def get_spatial_dims(self) -> List[List[int]]:
        """Get the spatial dimensions at each layer"""
        return self.spatial_dims.copy()

class AutoencoderLoss(nn.Module):
    """Composite loss function for autoencoder training"""
    def __init__(self, reconstruction_weight: float = 1.0,
                 feature_weight: float = 0.1):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.feature_weight = feature_weight

    def forward(self, input_data: torch.Tensor,
                reconstruction: torch.Tensor,
                embedding: torch.Tensor) -> torch.Tensor:
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, input_data)

        # Feature distribution loss (encourage normal distribution)
        feature_loss = torch.mean(torch.abs(embedding.mean(dim=0))) + \
                      torch.mean(torch.abs(embedding.std(dim=0) - 1))

        return self.reconstruction_weight * recon_loss + \
               self.feature_weight * feature_loss


class CNNFeatureExtractor(BaseFeatureExtractor):
    """CNN-based feature extractor implementation"""

    def _create_model(self) -> nn.Module:
        """Create CNN model"""
        return FeatureExtractorCNN(
            in_channels=self.config['dataset']['in_channels'],
            feature_dims=self.feature_dims
        ).to(self.device)

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch"""
        self.feature_extractor.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}',
                   unit='batch', leave=False)

        try:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.feature_extractor(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                batch_loss = running_loss / (batch_idx + 1)
                batch_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{batch_acc:.2f}%'
                })

                # Cleanup
                del inputs, outputs, loss
                if batch_idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            raise

        pbar.close()
        return running_loss / len(train_loader), 100. * correct / total

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.feature_extractor.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        try:
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.feature_extractor(inputs)
                    loss = F.cross_entropy(outputs, targets)

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # Cleanup
                    del inputs, outputs, loss

            return running_loss / len(val_loader), 100. * correct / total

        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise

    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from data"""
        self.feature_extractor.eval()
        features = []
        labels = []

        try:
            with torch.no_grad():
                for inputs, targets in tqdm(loader, desc="Extracting features"):
                    inputs = inputs.to(self.device)
                    outputs = self.feature_extractor(inputs)
                    features.append(outputs.cpu())
                    labels.append(targets)

                    # Cleanup
                    del inputs, outputs
                    if len(features) % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            return torch.cat(features), torch.cat(labels)

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def plot_confusion_matrix(self, loader: DataLoader, save_path: Optional[str] = None):
        """Plot confusion matrix of predictions"""
        self.feature_extractor.eval()
        all_preds = []
        all_labels = []

        try:
            with torch.no_grad():
                for inputs, targets in tqdm(loader, desc="Computing predictions"):
                    inputs = inputs.to(self.device)
                    outputs = self.feature_extractor(inputs)
                    _, preds = outputs.max(1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(targets.numpy())

                    # Cleanup
                    del inputs, outputs, preds

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Confusion matrix saved to {save_path}")
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise

    def get_feature_shape(self) -> Tuple[int, ...]:
        """Get shape of extracted features"""
        return (self.feature_dims,)

    def get_prediction_prob(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities for input"""
        self.feature_extractor.eval()
        with torch.no_grad():
            output = self.feature_extractor(input_tensor.to(self.device))
            return F.softmax(output, dim=1)

    def get_prediction(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Get prediction for input"""
        probs = self.get_prediction_prob(input_tensor)
        return probs.max(1)[1]

    def save_model(self, path: str):
        """Save model to path"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.feature_extractor.state_dict(),
            'config': self.config,
            'feature_dims': self.feature_dims
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from path"""
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.feature_extractor.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

class CNNFeatureExtractor(BaseFeatureExtractor):
    """CNN-based feature extractor implementation"""

    def _create_model(self) -> nn.Module:
        """Create CNN model"""
        return FeatureExtractorCNN(
            in_channels=self.config['dataset']['in_channels'],
            feature_dims=self.feature_dims
        ).to(self.device)

    def _load_from_checkpoint(self):
        """Load model from checkpoint"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Try to find latest checkpoint
        checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Load model state
                self.feature_extractor.load_state_dict(checkpoint['state_dict'])

                # Initialize and load optimizer
                self.optimizer = self._initialize_optimizer()
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("Optimizer state loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {str(e)}")

                # Load training state
                self.current_epoch = checkpoint.get('epoch', 0)
                self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))

                # Load history
                if 'history' in checkpoint:
                    self.history = defaultdict(list, checkpoint['history'])

                logger.info("Checkpoint loaded successfully")

            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                self.optimizer = self._initialize_optimizer()
        else:
            logger.info("No checkpoint found, starting from scratch")
            self.optimizer = self._initialize_optimizer()

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        if not os.path.exists(checkpoint_dir):
            return None

        dataset_name = self.config['dataset']['name']

        # Check for best model first
        best_path = os.path.join(checkpoint_dir, f"{dataset_name}_best.pth")
        if os.path.exists(best_path):
            return best_path

        # Check for latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            return checkpoint_path

        return None

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'state_dict': self.feature_extractor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'history': dict(self.history),
            'config': self.config
        }

        # Save latest checkpoint
        dataset_name = self.config['dataset']['name']
        filename = f"{dataset_name}_{'best' if is_best else 'checkpoint'}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved {'best' if is_best else 'latest'} checkpoint to {checkpoint_path}")

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch"""
        self.feature_extractor.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}',
                   unit='batch', leave=False)

        try:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.feature_extractor(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                batch_loss = running_loss / (batch_idx + 1)
                batch_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{batch_acc:.2f}%'
                })

                # Cleanup
                del inputs, outputs, loss
                if batch_idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            raise

        pbar.close()
        return running_loss / len(train_loader), 100. * correct / total

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.feature_extractor.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.feature_extractor(inputs)
                loss = F.cross_entropy(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Cleanup
                del inputs, outputs, loss

        return running_loss / len(val_loader), 100. * correct / total

    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from data"""
        self.feature_extractor.eval()
        features = []
        labels = []

        try:
            with torch.no_grad():
                for inputs, targets in tqdm(loader, desc="Extracting features"):
                    inputs = inputs.to(self.device)
                    outputs = self.feature_extractor(inputs)
                    features.append(outputs.cpu())
                    labels.append(targets)

                    # Cleanup
                    del inputs, outputs
                    if len(features) % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            return torch.cat(features), torch.cat(labels)

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def get_feature_shape(self) -> Tuple[int, ...]:
        """Get shape of extracted features"""
        return (self.feature_dims,)

    def plot_feature_distribution(self, loader: DataLoader, save_path: Optional[str] = None):
        """Plot distribution of extracted features"""
        features, _ = self.extract_features(loader)
        features = features.numpy()

        plt.figure(figsize=(12, 6))
        plt.hist(features.flatten(), bins=50, density=True)
        plt.title('Feature Distribution')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Feature distribution plot saved to {save_path}")
        plt.close()

class AutoEncoderFeatureExtractor(BaseFeatureExtractor):
    """Autoencoder-based feature extractor implementation"""

    def verify_config(self, config: Dict) -> Dict:
        """Add autoencoder-specific config verification"""
        config = super().verify_config(config)

        # Verify autoencoder-specific settings
        if 'autoencoder_config' not in config['model']:
            config['model']['autoencoder_config'] = {
                'reconstruction_weight': 1.0,
                'feature_weight': 0.1,
                'convergence_threshold': 0.001,
                'min_epochs': 10,
                'patience': 5
            }

        return config

    def _load_from_checkpoint(self):
        """Load model and training state from checkpoint"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Try to find latest checkpoint
        checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Load model state
                self.feature_extractor.load_state_dict(checkpoint['state_dict'])

                # Initialize and load optimizer
                self.optimizer = self._initialize_optimizer()
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("Optimizer state loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {str(e)}")

                # Load training state
                self.current_epoch = checkpoint.get('epoch', 0)
                self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))

                # Load history
                if 'history' in checkpoint:
                    self.history = defaultdict(list, checkpoint['history'])

                logger.info("Checkpoint loaded successfully")

            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                # Initialize fresh optimizer if checkpoint loading fails
                self.optimizer = self._initialize_optimizer()
        else:
            logger.info("No checkpoint found, starting from scratch")
            self.optimizer = self._initialize_optimizer()

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        if not os.path.exists(checkpoint_dir):
            return None

        dataset_name = self.config['dataset']['name']

        # Check for best model first
        best_path = os.path.join(checkpoint_dir, f"{dataset_name}_best.pth")
        if os.path.exists(best_path):
            return best_path

        # Check for latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            return checkpoint_path

        return None

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer based on configuration"""
        try:
            optimizer_config = self.config['model'].get('optimizer', {})

            # Set base parameters
            optimizer_params = {
                'lr': self.learning_rate,
                'weight_decay': optimizer_config.get('weight_decay', 1e-4)
            }

            # Configure optimizer-specific parameters
            optimizer_type = optimizer_config.get('type', 'Adam')
            if optimizer_type == 'SGD':
                optimizer_params['momentum'] = optimizer_config.get('momentum', 0.9)
                optimizer_params['nesterov'] = optimizer_config.get('nesterov', True)
            elif optimizer_type == 'Adam':
                optimizer_params['betas'] = (
                    optimizer_config.get('beta1', 0.9),
                    optimizer_config.get('beta2', 0.999)
                )
                optimizer_params['eps'] = optimizer_config.get('epsilon', 1e-8)
            elif optimizer_type == 'AdamW':
                optimizer_params['betas'] = (
                    optimizer_config.get('beta1', 0.9),
                    optimizer_config.get('beta2', 0.999)
                )
                optimizer_params['eps'] = optimizer_config.get('epsilon', 1e-8)

            # Get optimizer class
            try:
                optimizer_class = getattr(optim, optimizer_type)
            except AttributeError:
                logger.warning(f"Optimizer {optimizer_type} not found, using Adam")
                optimizer_class = optim.Adam
                optimizer_type = 'Adam'

            # Create optimizer
            optimizer = optimizer_class(self.feature_extractor.parameters(), **optimizer_params)

            logger.info(f"Initialized {optimizer_type} optimizer with parameters: {optimizer_params}")
            return optimizer

        except Exception as e:
            logger.error(f"Error initializing optimizer: {str(e)}")
            logger.info("Falling back to default Adam optimizer")
            return optim.Adam(self.feature_extractor.parameters(), lr=self.learning_rate)

    def _initialize_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler if specified in config"""
        scheduler_config = self.config['model'].get('scheduler', {})
        if not scheduler_config:
            return None

        scheduler_type = scheduler_config.get('type')
        if not scheduler_type:
            return None

        try:
            if scheduler_type == 'StepLR':
                return optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 7),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_type == 'ReduceLROnPlateau':
                return optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=scheduler_config.get('factor', 0.1),
                    patience=scheduler_config.get('patience', 10),
                    verbose=True
                )
            elif scheduler_type == 'CosineAnnealingLR':
                return optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.get('T_max', self.config['training']['epochs']),
                    eta_min=scheduler_config.get('eta_min', 0)
                )
        except Exception as e:
            logger.error(f"Error initializing scheduler: {str(e)}")
            return None

        return None

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """Train the feature extractor"""
        early_stopping = self.config['training'].get('early_stopping', {})
        patience = early_stopping.get('patience', 5)
        min_delta = early_stopping.get('min_delta', 0.001)
        max_epochs = self.config['training']['epochs']

        patience_counter = 0
        best_val_metric = float('inf')

        if not hasattr(self, 'training_start_time'):
            self.training_start_time = time.time()

        try:
            for epoch in range(self.current_epoch, max_epochs):
                self.current_epoch = epoch

                # Training
                train_loss, train_acc = self._train_epoch(train_loader)

                # Validation
                val_loss, val_acc = None, None
                if val_loader:
                    val_loss, val_acc = self._validate(val_loader)
                    current_metric = val_loss
                else:
                    current_metric = train_loss

                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    else:
                        self.scheduler.step()

                # Log metrics
                self.log_training_metrics(epoch, train_loss, train_acc, val_loss, val_acc,
                                       train_loader, val_loader)

                # Save checkpoint
                self._save_checkpoint(is_best=False)

                # Check for improvement
                if current_metric < best_val_metric - min_delta:
                    best_val_metric = current_metric
                    patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

                # Memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return self.history

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from data"""
        self.feature_extractor.eval()
        features = []
        labels = []

        try:
            with torch.no_grad():
                for inputs, targets in tqdm(loader, desc="Extracting features"):
                    inputs = inputs.to(self.device)
                    outputs = self.feature_extractor(inputs)
                    features.append(outputs.cpu())
                    labels.append(targets)

                    # Cleanup
                    del inputs, outputs
                    if len(features) % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            return torch.cat(features), torch.cat(labels)

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def log_training_metrics(self, epoch: int, train_loss: float, train_acc: float,
                            test_loss: Optional[float] = None, test_acc: Optional[float] = None,
                            train_loader: Optional[DataLoader] = None,
                            test_loader: Optional[DataLoader] = None):
        """Log training metrics"""
        # Calculate elapsed time
        elapsed_time = time.time() - self.training_start_time

        # Prepare metrics dictionary
        metrics = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_time': elapsed_time,
            'elapsed_time_formatted': str(timedelta(seconds=int(elapsed_time))),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_samples': len(train_loader.dataset) if train_loader else None,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_samples': len(test_loader.dataset) if test_loader else None,
        }

        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        if test_loss is not None:
            self.history['val_loss'].append(test_loss)
        if test_acc is not None:
            self.history['val_acc'].append(test_acc)

        # Add to training log
        self.training_log.append(metrics)

        # Save to CSV
        log_df = pd.DataFrame(self.training_log)
        log_path = os.path.join(self.log_dir, 'training_metrics.csv')
        log_df.to_csv(log_path, index=False)

        # Log to console
        log_message = (f"Epoch {epoch + 1}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if test_loss is not None:
            log_message += f", Val Loss: {test_loss:.4f}, Val Acc: {test_acc:.2f}%"
        logger.info(log_message)

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if not self.history:
            logger.warning("No training history available to plot")
            return

        plt.figure(figsize=(15, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        if 'val_acc' in self.history:
            plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        plt.close()

    def save_features(self, features: torch.Tensor, labels: torch.Tensor, output_path: str):
        """Save extracted features to CSV"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Process in chunks to manage memory
            chunk_size = 1000
            total_samples = features.shape[0]

            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)

                feature_dict = {
                    f'feature_{i}': features[start_idx:end_idx, i].numpy()
                    for i in range(features.shape[1])
                }
                feature_dict['target'] = labels[start_idx:end_idx].numpy()

                df = pd.DataFrame(feature_dict)
                mode = 'w' if start_idx == 0 else 'a'
                header = start_idx == 0

                df.to_csv(output_path, mode=mode, index=False, header=header)

                # Cleanup
                del feature_dict, df
                gc.collect()

            logger.info(f"Features saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise


    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'state_dict': self.feature_extractor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'history': dict(self.history),
            'config': self.config
        }

        # Save latest checkpoint
        dataset_name = self.config['dataset']['name']
        filename = f"{dataset_name}_{'best' if is_best else 'checkpoint'}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved {'best' if is_best else 'latest'} checkpoint to {checkpoint_path}")

    def _create_model(self) -> nn.Module:
        """Create autoencoder model"""
        input_shape = (self.config['dataset']['in_channels'],
                      *self.config['dataset']['input_size'])
        return DynamicAutoencoder(
            input_shape=input_shape,
            feature_dims=self.feature_dims
        ).to(self.device)

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch"""
        self.feature_extractor.train()
        running_loss = 0.0
        reconstruction_accuracy = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}',
                   unit='batch', leave=False)

        for batch_idx, (inputs, _) in enumerate(pbar):
            try:
                inputs = inputs.to(self.device)

                self.optimizer.zero_grad()
                embedding, reconstruction = self.feature_extractor(inputs)

                # Calculate loss
                ae_config = self.config['model']['autoencoder_config']
                loss = AutoencoderLoss(
                    reconstruction_weight=ae_config['reconstruction_weight'],
                    feature_weight=ae_config['feature_weight']
                )(inputs, reconstruction, embedding)

                loss.backward()
                self.optimizer.step()

                # Update metrics
                running_loss += loss.item()
                reconstruction_accuracy += 1.0 - F.mse_loss(reconstruction, inputs).item()

                # Update progress bar
                batch_loss = running_loss / (batch_idx + 1)
                batch_acc = (reconstruction_accuracy / (batch_idx + 1)) * 100
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'recon_acc': f'{batch_acc:.2f}%'
                })

                # Cleanup
                del inputs, embedding, reconstruction, loss
                if batch_idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                raise

        pbar.close()
        return (running_loss / len(train_loader),
                (reconstruction_accuracy / len(train_loader)) * 100)

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.feature_extractor.eval()
        running_loss = 0.0
        reconstruction_accuracy = 0.0

        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(self.device)
                embedding, reconstruction = self.feature_extractor(inputs)

                ae_config = self.config['model']['autoencoder_config']
                loss = AutoencoderLoss(
                    reconstruction_weight=ae_config['reconstruction_weight'],
                    feature_weight=ae_config['feature_weight']
                )(inputs, reconstruction, embedding)

                running_loss += loss.item()
                reconstruction_accuracy += 1.0 - F.mse_loss(reconstruction, inputs).item()

                del inputs, embedding, reconstruction, loss

        return (running_loss / len(val_loader),
                (reconstruction_accuracy / len(val_loader)) * 100)

    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override to use autoencoder's encoding"""
        self.feature_extractor.eval()
        features = []
        labels = []

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Extracting features"):
                inputs = inputs.to(self.device)
                # Use encode method directly instead of forward
                embedding = self.feature_extractor.encode(inputs)
                features.append(embedding.cpu())
                labels.append(targets)

                del inputs, embedding

        return torch.cat(features), torch.cat(labels)

    def get_reconstruction(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Get reconstruction for input tensor"""
        self.feature_extractor.eval()
        with torch.no_grad():
            embedding, reconstruction = self.feature_extractor(input_tensor)
            return reconstruction

    def get_reconstruction_error(self, input_tensor: torch.Tensor) -> float:
        """Calculate reconstruction error for input tensor"""
        reconstruction = self.get_reconstruction(input_tensor)
        return F.mse_loss(reconstruction, input_tensor).item()

    def visualize_reconstructions(self, dataloader: DataLoader, num_samples: int = 8,
                                save_path: Optional[str] = None):
        """Visualize original and reconstructed images"""
        self.feature_extractor.eval()

        # Get samples
        images, _ = next(iter(dataloader))
        images = images[:num_samples].to(self.device)

        with torch.no_grad():
            _, reconstructions = self.feature_extractor(images)

        # Plot results
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))

        for i in range(num_samples):
            # Original
            axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')

            # Reconstruction
            axes[1, i].imshow(reconstructions[i].cpu().permute(1, 2, 0))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Reconstruction visualization saved to {save_path}")
        plt.close()

    def plot_latent_space(self, dataloader: DataLoader, num_samples: int = 1000,
                         save_path: Optional[str] = None):
        """Plot 2D visualization of latent space"""
        if self.feature_dims < 2:
            logger.warning("Latent space dimension too small for visualization")
            return

        self.feature_extractor.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                if len(embeddings) * inputs.size(0) >= num_samples:
                    break

                inputs = inputs.to(self.device)
                embedding = self.feature_extractor.encode(inputs)
                embeddings.append(embedding.cpu())
                labels.extend(targets.tolist())

        embeddings = torch.cat(embeddings, dim=0)[:num_samples]
        labels = labels[:num_samples]

        # Use PCA for visualization if dimensions > 2
        if self.feature_dims > 2:
            from sklearn.decomposition import PCA
            embeddings = PCA(n_components=2).fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title('Latent Space Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

class FeatureExtractorFactory:
    """Factory class for creating feature extractors"""

    @staticmethod
    def create(config: Dict, device: Optional[str] = None) -> BaseFeatureExtractor:
        """
        Create appropriate feature extractor based on configuration.

        Args:
            config: Configuration dictionary
            device: Optional device specification

        Returns:
            Instance of appropriate feature extractor
        """
        encoder_type = config['model'].get('encoder_type', 'cnn').lower()

        if encoder_type == 'cnn':
            return CNNFeatureExtractor(config, device)
        elif encoder_type == 'autoenc':
            return AutoEncoderFeatureExtractor(config, device)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

def get_feature_extractor(config: Dict, device: Optional[str] = None) -> BaseFeatureExtractor:
    """Convenience function to create feature extractor"""
    return FeatureExtractorFactory.create(config, device)


class CustomImageDataset(Dataset):
    """Custom dataset for loading images from directory structure"""
    def __init__(self, data_dir: str, transform=None, csv_file: Optional[str] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.label_encoder = {}
        self.reverse_encoder = {}

        if csv_file and os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file)
        else:
            self.image_files = []
            self.labels = []
            unique_labels = sorted(os.listdir(data_dir))

            for idx, label in enumerate(unique_labels):
                self.label_encoder[label] = idx
                self.reverse_encoder[idx] = label

            # Save label encodings
            encoding_file = os.path.join(data_dir, 'label_encodings.json')
            with open(encoding_file, 'w') as f:
                json.dump({
                    'label_to_id': self.label_encoder,
                    'id_to_label': self.reverse_encoder
                }, f, indent=4)

            for class_name in unique_labels:
                class_dir = os.path.join(data_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            self.image_files.append(os.path.join(class_dir, img_name))
                            self.labels.append(self.label_encoder[class_name])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class DatasetProcessor:
    """Processor for handling both torchvision and custom datasets"""
    SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    def __init__(self, datafile: str = "MNIST", datatype: str = "torchvision",
                 output_dir: str = "data"):
        self.datafile = datafile
        self.datatype = datatype.lower()
        self.output_dir = output_dir

        # Set dataset name
        if self.datatype == 'torchvision':
            self.dataset_name = self.datafile.lower()
        else:
            self.dataset_name = os.path.basename(os.path.normpath(datafile))

        # Create dataset-specific directories
        self.dataset_dir = os.path.join(output_dir, self.dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Set paths for configuration files
        self.config_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.json")
        self.conf_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.conf")
        self.dbnn_conf_path = os.path.join(self.dataset_dir, "adaptive_dbnn.conf")

    def get_transforms(self, config: Dict, is_train: bool = True) -> transforms.Compose:
        """Get transforms based on configuration"""
        transform_list = []

        # Handle dataset-specific transforms
        if config['dataset']['name'].upper() == 'MNIST':
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        # Get augmentation config
        aug_config = config.get('augmentation', {})
        if not aug_config.get('enabled', True):
            transform_list.append(transforms.ToTensor())
            return transforms.Compose(transform_list)

        # Basic transforms
        image_size = config['dataset']['input_size']
        transform_list.append(transforms.Resize(image_size))

        if is_train:
            # Training augmentations
            if aug_config.get('random_crop', {}).get('enabled', False):
                transform_list.append(transforms.RandomCrop(image_size, padding=4))
            if aug_config.get('horizontal_flip', {}).get('enabled', False):
                transform_list.append(transforms.RandomHorizontalFlip())
            if aug_config.get('color_jitter', {}).get('enabled', False):
                transform_list.append(transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ))

        # Normalization
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['dataset']['mean'],
                std=config['dataset']['std']
            )
        ])

        return transforms.Compose(transform_list)

    def _generate_dataset_conf(self, feature_dims: int) -> Dict:
        """Generate dataset-specific configuration"""
        dataset_conf = {
            "file_path": f"{self.dataset_name}.csv",
            "_path_comment": "Dataset file path",
            "column_names": [f"feature_{i}" for i in range(feature_dims)] + ["target"],
            "_columns_comment": "Feature and target column names",
            "separator": ",",
            "_separator_comment": "CSV separator character",
            "has_header": True,
            "_header_comment": "Has header row",
            "target_column": "target",
            "_target_comment": "Target column name",
            "likelihood_config": {
                "feature_group_size": 2,
                "_group_comment": "Feature group size for analysis",
                "max_combinations": min(1000, feature_dims * (feature_dims - 1) // 2),
                "_combinations_comment": "Maximum feature combinations to analyze",
                "bin_sizes": [20],
                "_bins_comment": "Histogram bin sizes"
            },
            "active_learning": {
                "tolerance": 1.0,
                "_tolerance_comment": "Learning tolerance",
                "cardinality_threshold_percentile": 95,
                "_percentile_comment": "Cardinality threshold percentile",
                "strong_margin_threshold": 0.3,
                "_strong_comment": "Strong margin threshold",
                "marginal_margin_threshold": 0.1,
                "_marginal_comment": "Marginal margin threshold",
                "min_divergence": 0.1,
                "_divergence_comment": "Minimum divergence threshold"
            },
            "training_params": {
                "Save_training_epochs": True,
                "_save_comment": "Whether to save training epochs",
                "training_save_path": os.path.join(self.dataset_dir, "training_data"),
                "_save_path_comment": "Path to save training data"
            },
            "modelType": "Histogram",
            "_model_comment": "Model type (Histogram or Gaussian)"
        }
        return dataset_conf

    def _generate_main_config(self, train_dir: str) -> Dict:
        """
        Generate main configuration file with all necessary parameters.

        Args:
            train_dir: Path to training data directory

        Returns:
            Dict: Complete configuration dictionary
        """
        try:
            # Detect image properties from first image
            input_size, in_channels = self._detect_image_properties(train_dir)

            # Count classes
            class_dirs = [d for d in os.listdir(train_dir)
                         if os.path.isdir(os.path.join(train_dir, d))]
            num_classes = len(class_dirs)

            if num_classes == 0:
                raise ValueError(f"No class directories found in {train_dir}")

            # Set appropriate normalization values
            if in_channels == 1:  # Grayscale
                mean = [0.5]
                std = [0.5]
            else:  # RGB
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

            # Calculate appropriate feature dimensions
            feature_dims = min(128, np.prod(input_size) // 4)  # Reasonable compression

            config = {
                "dataset": {
                    "name": self.dataset_name,
                    "type": self.datatype,
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                    "input_size": list(input_size),
                    "mean": mean,
                    "std": std,
                    "train_dir": train_dir,
                    "test_dir": os.path.join(os.path.dirname(train_dir), 'test')
                },

                "model": {
                    "encoder_type": "cnn",  # Default to CNN
                    "feature_dims": feature_dims,
                    "learning_rate": 0.001,

                    "optimizer": {
                        "type": "Adam",
                        "weight_decay": 1e-4,
                        "momentum": 0.9,
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "epsilon": 1e-8
                    },

                    "scheduler": {
                        "type": "ReduceLROnPlateau",
                        "factor": 0.1,
                        "patience": 10,
                        "min_lr": 1e-6,
                        "verbose": True
                    },

                    "autoencoder_config": {
                        "reconstruction_weight": 1.0,
                        "feature_weight": 0.1,
                        "convergence_threshold": 0.001,
                        "min_epochs": 10,
                        "patience": 5
                    }
                },

                "training": {
                    "batch_size": 32,
                    "epochs": 20,
                    "num_workers": min(4, os.cpu_count() or 1),
                    "checkpoint_dir": os.path.join(self.dataset_dir, "checkpoints"),
                    "validation_split": 0.2,

                    "early_stopping": {
                        "patience": 5,
                        "min_delta": 0.001
                    }
                },

                "execution_flags": {
                    "mode": "train_and_predict",
                    "use_gpu": torch.cuda.is_available(),
                    "mixed_precision": True,
                    "distributed_training": False,
                    "debug_mode": False,
                    "use_previous_model": True,
                    "fresh_start": False
                },

                "augmentation": {
                    "enabled": True,
                    "random_crop": {
                        "enabled": True,
                        "padding": 4
                    },
                    "random_rotation": {
                        "enabled": True,
                        "degrees": 10
                    },
                    "horizontal_flip": {
                        "enabled": True,
                        "probability": 0.5
                    },
                    "vertical_flip": {
                        "enabled": False
                    },
                    "color_jitter": {
                        "enabled": True,
                        "brightness": 0.2,
                        "contrast": 0.2,
                        "saturation": 0.2,
                        "hue": 0.1
                    },
                    "normalize": {
                        "enabled": True,
                        "mean": mean,
                        "std": std
                    }
                },

                "logging": {
                    "log_dir": os.path.join(self.dataset_dir, "logs"),
                    "tensorboard": {
                        "enabled": True,
                        "log_dir": os.path.join(self.dataset_dir, "tensorboard")
                    },
                    "save_frequency": 5,  # Save every N epochs
                    "metrics": ["loss", "accuracy", "reconstruction_error"]
                },

                "output": {
                    "features_file": os.path.join(self.dataset_dir, f"{self.dataset_name}.csv"),
                    "model_dir": os.path.join(self.dataset_dir, "models"),
                    "visualization_dir": os.path.join(self.dataset_dir, "visualizations")
                }
            }

            # Create necessary directories
            os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
            os.makedirs(config['logging']['log_dir'], exist_ok=True)
            os.makedirs(config['logging']['tensorboard']['log_dir'], exist_ok=True)
            os.makedirs(config['output']['model_dir'], exist_ok=True)
            os.makedirs(config['output']['visualization_dir'], exist_ok=True)

            logger.info(f"Generated configuration for dataset: {self.dataset_name}")
            logger.info(f"Input shape: {in_channels}x{input_size[0]}x{input_size[1]}")
            logger.info(f"Number of classes: {num_classes}")
            logger.info(f"Feature dimensions: {feature_dims}")

            return config

        except Exception as e:
            logger.error(f"Error generating main configuration: {str(e)}")
            raise

    def generate_default_config(self, train_dir: str) -> Dict:
        """Generate default configuration and all necessary conf files"""
        try:
            # Generate main configuration
            config = self._generate_main_config(train_dir)

            # Save main JSON config
            os.makedirs(self.dataset_dir, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)

            # Generate and save dataset-specific .conf
            dataset_conf = self._generate_dataset_conf(config['model']['feature_dims'])
            with open(self.conf_path, 'w') as f:
                json.dump(dataset_conf, f, indent=4)

            # Generate and save adaptive_dbnn.conf
            dbnn_config = self._generate_dbnn_config(config)
            with open(self.dbnn_conf_path, 'w') as f:
                json.dump(dbnn_config, f, indent=4)

            logger.info(f"Configuration files generated in: {self.dataset_dir}")
            logger.info(f"  - Main config: {os.path.basename(self.config_path)}")
            logger.info(f"  - Dataset conf: {os.path.basename(self.conf_path)}")
            logger.info(f"  - DBNN conf: {os.path.basename(self.dbnn_conf_path)}")

            return config

        except Exception as e:
            logger.error(f"Error generating configuration files: {str(e)}")
            raise

    def _generate_dbnn_config(self, main_config: Dict) -> Dict:
        """Generate DBNN-specific configuration"""
        return {
            "training_params": {
                "trials": main_config['training']['epochs'],
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "learning_rate": main_config['model']['learning_rate'],
                "random_seed": 42,
                "epochs": main_config['training']['epochs'],
                "test_fraction": 0.2,
                "enable_adaptive": True,
                "modelType": "Histogram",
                "compute_device": "auto",
                "Save_training_epochs": True,
                "training_save_path": os.path.join(self.dataset_dir, "training_data")
            },
            "execution_flags": {
                "train": True,
                "train_only": False,
                "predict": True,
                "gen_samples": False,
                "fresh_start": False,
                "use_previous_model": True
            }
        }


    def _detect_image_properties(self, folder_path: str) -> Tuple[Tuple[int, int], int]:
        """
        Detect image size and number of channels from dataset.

        Args:
            folder_path: Path to folder containing images

        Returns:
            Tuple containing ((width, height), channels)
        """
        size_counts = defaultdict(int)
        channel_counts = defaultdict(int)
        samples_checked = 0

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS):
                    try:
                        with Image.open(os.path.join(root, file)) as img:
                            # Convert to tensor to handle channels properly
                            tensor = transforms.ToTensor()(img)
                            height, width = tensor.shape[1], tensor.shape[2]
                            channels = tensor.shape[0]

                            size_counts[(width, height)] += 1
                            channel_counts[channels] += 1
                            samples_checked += 1

                            # Stop after checking enough samples
                            if samples_checked >= 50:
                                break
                    except Exception as e:
                        logger.warning(f"Could not process image {file}: {str(e)}")
                        continue

            if samples_checked >= 50:
                break

        if not size_counts:
            raise ValueError(f"No valid images found in {folder_path}")

        # Get most common dimensions and channels
        input_size = max(size_counts.items(), key=lambda x: x[1])[0]
        in_channels = max(channel_counts.items(), key=lambda x: x[1])[0]

        return input_size, in_channels


    def process(self) -> Tuple[str, Optional[str]]:
        """Process dataset and return paths to train and test directories"""
        if self.datatype == 'torchvision':
            return self._process_torchvision()
        else:
            return self._process_custom()

    def _process_torchvision(self) -> Tuple[str, str]:
        """Process torchvision dataset"""
        dataset_name = self.datafile.upper()
        if not hasattr(datasets, dataset_name):
            raise ValueError(f"Torchvision dataset {dataset_name} not found")

        # Setup paths in dataset-specific directory
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Download and process datasets
        transform = transforms.ToTensor()

        train_dataset = getattr(datasets, dataset_name)(
            root=self.output_dir,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = getattr(datasets, dataset_name)(
            root=self.output_dir,
            train=False,
            download=True,
            transform=transform
        )

        # Save images with class directories
        def save_dataset_images(dataset, output_dir, split_name):
            logger.info(f"Processing {split_name} split...")

            class_to_idx = getattr(dataset, 'class_to_idx', None)
            if class_to_idx:
                idx_to_class = {v: k for k, v in class_to_idx.items()}

            with tqdm(total=len(dataset), desc=f"Saving {split_name} images") as pbar:
                for idx, (img, label) in enumerate(dataset):
                    class_name = idx_to_class[label] if class_to_idx else str(label)
                    class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)

                    if isinstance(img, torch.Tensor):
                        img = transforms.ToPILImage()(img)

                    img_path = os.path.join(class_dir, f"{idx}.png")
                    img.save(img_path)
                    pbar.update(1)

        save_dataset_images(train_dataset, train_dir, "training")
        save_dataset_images(test_dataset, test_dir, "test")

        return train_dir, test_dir

    def _process_custom(self) -> Tuple[str, Optional[str]]:
        """Process custom dataset"""
        if not os.path.exists(self.datafile):
            raise ValueError(f"Dataset path not found: {self.datafile}")

        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        # Check if dataset already has train/test structure
        if os.path.isdir(os.path.join(self.datafile, "train")) and \
           os.path.isdir(os.path.join(self.datafile, "test")):
            # Copy existing structure
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

            shutil.copytree(os.path.join(self.datafile, "train"), train_dir)
            shutil.copytree(os.path.join(self.datafile, "test"), test_dir)

            return train_dir, test_dir

        # Handle single directory with class subdirectories
        if not os.path.isdir(self.datafile):
            raise ValueError(f"Invalid dataset path: {self.datafile}")

        class_dirs = [d for d in os.listdir(self.datafile)
                     if os.path.isdir(os.path.join(self.datafile, d))]

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.datafile}")

        # Ask user about train/test split
        response = input("Create train/test split? (y/n): ").lower()
        if response == 'y':
            test_size = float(input("Enter test size (0-1, default: 0.2): ") or "0.2")
            return self._create_train_test_split(self.datafile, test_size)
        else:
            # Use all data for training
            os.makedirs(train_dir, exist_ok=True)
            for class_dir in class_dirs:
                src = os.path.join(self.datafile, class_dir)
                dst = os.path.join(train_dir, class_dir)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            return train_dir, None

    def _create_train_test_split(self, source_dir: str, test_size: float) -> Tuple[str, str]:
        """Create train/test split from source directory"""
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for class_name in tqdm(os.listdir(source_dir), desc="Processing classes"):
            class_path = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            # Create class directories
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # Get all image files
            image_files = [f for f in os.listdir(class_path)
                         if f.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS)]

            # Random split
            random.shuffle(image_files)
            split_idx = int((1 - test_size) * len(image_files))
            train_files = image_files[:split_idx]
            test_files = image_files[split_idx:]

            # Copy files
            for fname in train_files:
                shutil.copy2(
                    os.path.join(class_path, fname),
                    os.path.join(train_class_dir, fname)
                )

            for fname in test_files:
                shutil.copy2(
                    os.path.join(class_path, fname),
                    os.path.join(test_class_dir, fname)
                )

        return train_dir, test_dir

class ConfigManager:
    """Manages configuration generation and validation"""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _detect_image_properties(self, folder_path: str) -> Tuple[Tuple[int, int], int]:
        """Detect image size and channels from dataset"""
        img_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        size_counts = defaultdict(int)
        channel_counts = defaultdict(int)

        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in img_formats):
                    try:
                        with Image.open(os.path.join(root, file)) as img:
                            tensor = transforms.ToTensor()(img)
                            height, width = tensor.shape[1], tensor.shape[2]
                            channels = tensor.shape[0]

                            size_counts[(width, height)] += 1
                            channel_counts[channels] += 1
                    except Exception as e:
                        logger.warning(f"Could not read image {file}: {str(e)}")
                        continue

            if sum(size_counts.values()) >= 50:
                break

        if not size_counts:
            raise ValueError(f"No valid images found in {folder_path}")

        input_size = max(size_counts, key=size_counts.get)
        in_channels = max(channel_counts, key=channel_counts.get)

        return input_size, in_channels

def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    return logger

def get_dataset(config: Dict, transform) -> Tuple[Dataset, Optional[Dataset]]:
    """Get dataset based on configuration"""
    dataset_config = config['dataset']

    if dataset_config['type'] == 'torchvision':
        train_dataset = getattr(torchvision.datasets, dataset_config['name'].upper())(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = getattr(torchvision.datasets, dataset_config['name'].upper())(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    else:
        train_dir = dataset_config['train_dir']
        test_dir = dataset_config.get('test_dir')

        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")

        train_dataset = CustomImageDataset(
            data_dir=train_dir,
            transform=transform
        )

        test_dataset = None
        if test_dir and os.path.exists(test_dir):
            test_dataset = CustomImageDataset(
                data_dir=test_dir,
                transform=transform
            )

    if config['training'].get('merge_datasets', False) and test_dataset is not None:
        return CombinedDataset(train_dataset, test_dataset), None

    return train_dataset, test_dataset

class CombinedDataset(Dataset):
    """Dataset that combines train and test sets"""
    def __init__(self, train_dataset: Dataset, test_dataset: Dataset):
        self.combined_data = ConcatDataset([train_dataset, test_dataset])

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        return self.combined_data[idx]

def update_config_with_args(config: Dict, args) -> Dict:
    """Update configuration with command line arguments"""
    if hasattr(args, 'encoder_type'):
        config['model']['encoder_type'] = args.encoder_type
    if hasattr(args, 'batch_size'):
        config['training']['batch_size'] = args.batch_size
    if hasattr(args, 'epochs'):
        config['training']['epochs'] = args.epochs
    if hasattr(args, 'workers'):
        config['training']['num_workers'] = args.workers
    if hasattr(args, 'learning_rate'):
        config['model']['learning_rate'] = args.learning_rate
    if hasattr(args, 'cpu'):
        config['execution_flags']['use_gpu'] = not args.cpu
    if hasattr(args, 'debug'):
        config['execution_flags']['debug_mode'] = args.debug

    return config

def print_usage():
    """Print usage information with examples"""
    print("\nCDBNN (Convolutional Deep Bayesian Neural Network) Image Processor")
    print("=" * 80)
    print("\nUsage:")
    print("  1. Interactive Mode:")
    print("     python cdbnn.py")
    print("\n  2. Command Line Mode:")
    print("     python cdbnn.py --data_type TYPE --data PATH [options]")

    print("\nRequired Arguments:")
    print("  --data_type     Type of dataset ('torchvision' or 'custom')")
    print("  --data          Dataset name (for torchvision) or path (for custom)")

    print("\nOptional Arguments:")
    print("  --encoder_type  Type of encoder ('cnn' or 'autoenc')")
    print("  --config        Path to configuration file (overrides other options)")
    print("  --batch_size    Batch size for training (default: 32)")
    print("  --epochs        Number of training epochs (default: 20)")
    print("  --workers       Number of data loading workers (default: 4)")
    print("  --learning_rate Learning rate (default: 0.001)")
    print("  --output-dir    Output directory (default: data)")
    print("  --cpu          Force CPU usage even if GPU is available")
    print("  --debug        Enable debug mode with verbose logging")

    print("\nExamples:")
    print("  1. Process MNIST dataset using CNN:")
    print("     python cdbnn.py --data_type torchvision --data MNIST --encoder_type cnn")

    print("  2. Process custom dataset using Autoencoder:")
    print("     python cdbnn.py --data_type custom --data path/to/images --encoder_type autoenc")

def parse_arguments():
    """Parse command line arguments"""
    if len(sys.argv) == 1:
        return None

    parser = argparse.ArgumentParser(description='CNN/Autoencoder Feature Extractor')

    # Required arguments
    parser.add_argument('--data_type', type=str, choices=['torchvision', 'custom'],
                      help='type of dataset (torchvision or custom)')
    parser.add_argument('--data', type=str,
                      help='dataset name for torchvision or path for custom dataset')

    # Optional arguments
    parser.add_argument('--encoder_type', type=str, choices=['cnn', 'autoenc'],
                      default='cnn', help='type of encoder (default: cnn)')
    parser.add_argument('--config', type=str,
                      help='path to configuration file')
    parser.add_argument('--batch_size', type=int,
                      help='batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int,
                      help='number of training epochs (default: 20)')
    parser.add_argument('--workers', type=int,
                      help='number of data loading workers (default: 4)')
    parser.add_argument('--learning_rate', type=float,
                      help='learning rate (default: 0.001)')
    parser.add_argument('--output-dir', type=str, default='data',
                      help='output directory (default: data)')
    parser.add_argument('--cpu', action='store_true',
                      help='force CPU usage')
    parser.add_argument('--debug', action='store_true',
                      help='enable debug mode')

    return parser.parse_args()

def get_interactive_args():
    """Get arguments interactively"""
    args = argparse.Namespace()

    # Get data type
    while True:
        data_type = input("\nEnter dataset type (torchvision/custom): ").strip().lower()
        if data_type in ['torchvision', 'custom']:
            args.data_type = data_type
            break
        print("Invalid type. Please enter 'torchvision' or 'custom'")

    # Get data path/name
    args.data = input("Enter dataset name (torchvision) or path (custom): ").strip()

    # Get encoder type
    while True:
        encoder_type = input("Enter encoder type (cnn/autoenc) [default: cnn]: ").strip().lower()
        if not encoder_type:
            encoder_type = 'cnn'
        if encoder_type in ['cnn', 'autoenc']:
            args.encoder_type = encoder_type
            break
        print("Invalid encoder type. Please enter 'cnn' or 'autoenc'")

    # Get optional parameters
    args.batch_size = int(input("Enter batch size (default: 32): ").strip() or "32")
    args.epochs = int(input("Enter number of epochs (default: 20): ").strip() or "20")
    args.output_dir = input("Enter output directory (default: data): ").strip() or "data"

    # Set defaults for other arguments
    args.workers = 4
    args.learning_rate = 0.01
    args.cpu = False
    args.debug = False
    args.config = None

    return args

def main():
    """Main execution function"""
    try:
        # Set up logging
        logger = setup_logging()
        logger.info("Starting feature extraction process...")

        # Get arguments
        args = parse_arguments()
        config = None

        # Handle interactive mode if no arguments provided
        if args is None:
            print("\nEntering interactive mode...")
            args = get_interactive_args()

        # Load configuration if provided
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            with open(args.config, 'r') as f:
                config = json.load(f)

        # Initialize processor
        processor = DatasetProcessor(
            datafile=args.data,
            datatype=args.data_type.lower(),
            output_dir=args.output_dir
        )

        # Process dataset
        logger.info("Processing dataset...")
        train_dir, test_dir = processor.process()
        logger.info(f"Dataset processed: train_dir={train_dir}, test_dir={test_dir}")

        # Generate or update configuration
        if not config:
            logger.info("Generating configuration...")
            config = processor.generate_default_config(train_dir)

        # Update config with command line arguments
        config = update_config_with_args(config, args)

        # Get transforms
        transform = processor.get_transforms(config)

        # Prepare datasets
        train_dataset, test_dataset = get_dataset(config, transform)
        if train_dataset is None:
            raise ValueError("No training dataset available")
        num_workers=config['training']['num_workers']

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers']
        )

        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['training']['num_workers']
            )

        # Create feature extractor
        logger.info(f"Initializing {config['model']['encoder_type']} feature extractor...")
        feature_extractor = get_feature_extractor(config)

        # Train model
        logger.info("Starting model training...")
        history = feature_extractor.train(train_loader, test_loader)

        # Extract features
        logger.info("Extracting features...")
        train_features, train_labels = feature_extractor.extract_features(train_loader)

        if test_loader:
            test_features, test_labels = feature_extractor.extract_features(test_loader)
            features = torch.cat([train_features, test_features])
            labels = torch.cat([train_labels, test_labels])
        else:
            features = train_features
            labels = train_labels

        # Save features
        output_path = os.path.join(args.output_dir, config['dataset']['name'],
                                f"{config['dataset']['name']}.csv")
        feature_extractor.save_features(features, labels, output_path)
        logger.info(f"Features saved to {output_path}")

        # Plot training history
        if history:
            plot_path = os.path.join(args.output_dir, config['dataset']['name'],
                                  'training_history.png')
            feature_extractor.plot_training_history(plot_path)

        logger.info("Processing completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        if args.debug:
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
