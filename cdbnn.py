import torch
import copy  #
import  argparse
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
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
    SUPPORTED_COMPRESSION_FORMATS = ('.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2')

    def __init__(self, datafile="MNIST", datatype="torchvision", output_dir="data"):
        self.datafile = datafile
        self.datatype = datatype
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Check for existing config and data files
        self.config_path = os.path.join(output_dir, f"{datafile}.json")
        self.conf_path = os.path.join(output_dir, f"{datafile}.conf")
        self.csv_path = os.path.join(output_dir, f"{datafile}.csv")

        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = None

    def check_existing_files(self):
        """Check for existing files and ask for user consent before overwriting"""
        files_exist = all(os.path.exists(f) for f in [self.config_path, self.conf_path, self.csv_path])

        if files_exist:
            response = input("Dataset files already exist. Do you want to regenerate them? (y/n): ")
            return response.lower() != 'y'
        return False


    @staticmethod
    def verify_config(config: dict) -> bool:
        """Verify configuration has all required fields"""
        required_fields = {
            'dataset': ['name', 'type', 'in_channels', 'num_classes', 'input_size', 'mean', 'std'],
            'model': ['feature_dims', 'learning_rate'],
            'training': ['batch_size', 'epochs', 'num_workers'],
            'execution_flags': ['mode', 'use_gpu']
        }

        for section, fields in required_fields.items():
            if section not in config:
                print(f"Missing section: {section}")
                return False
            for field in fields:
                if field not in config[section]:
                    print(f"Missing field: {section}.{field}")
                    return False
        return True
    def get_transforms(self, config: Dict, is_train: bool = True) -> transforms.Compose:
        """Get transforms based on configuration"""
        transform_list = []

        # Handle dataset-specific transforms
        if 'dataset' in config:
            if config['dataset']['name'].upper() == 'MNIST':
                transform_list.append(transforms.Grayscale(num_output_channels=1))

        # Get augmentation config
        aug_config = config.get('augmentation', {})
        if not aug_config.get('enabled', True):
            transform_list.append(transforms.ToTensor())
            return transforms.Compose(transform_list)

        components = aug_config.get('components', {})
        image_size = config['dataset']['input_size']
        min_dim = min(image_size[0], image_size[1])

        # Resize
        if components.get('resize', {}).get('enabled', True):
            transform_list.append(transforms.Resize(image_size))

        if is_train:
            # Random crop
            if components.get('random_crop', {}).get('enabled', False):
                crop_config = components['random_crop']
                crop_size = crop_config.get('size', image_size)
                padding = crop_config.get('padding', 0)
                if isinstance(crop_size, list):
                    crop_size = tuple(crop_size)
                transform_list.append(transforms.RandomCrop(crop_size, padding=padding))

            # Horizontal flip
            if components.get('horizontal_flip', {}).get('enabled', False):
                probability = components['horizontal_flip'].get('probability', 0.5)
                transform_list.append(transforms.RandomHorizontalFlip(p=probability))

            # Vertical flip
            if components.get('vertical_flip', {}).get('enabled', False):
                transform_list.append(transforms.RandomVerticalFlip())

            # Random rotation
            if components.get('random_rotation', {}).get('enabled', False):
                degrees = components['random_rotation'].get('degrees', 15)
                transform_list.append(transforms.RandomRotation(degrees))

            # Color jitter
            if components.get('color_jitter', {}).get('enabled', False):
                color_jitter = components['color_jitter']
                transform_list.append(transforms.ColorJitter(
                    brightness=color_jitter.get('brightness', 0),
                    contrast=color_jitter.get('contrast', 0),
                    saturation=color_jitter.get('saturation', 0),
                    hue=color_jitter.get('hue', 0)
                ))
        else:
            # Center crop for validation/test
            if components.get('center_crop', {}).get('enabled', False):
                crop_config = components['center_crop']
                crop_size = crop_config.get('size', image_size)
                if isinstance(crop_size, list):
                    crop_size = tuple(crop_size)
                transform_list.append(transforms.CenterCrop(crop_size))

        # ToTensor should always be included
        transform_list.append(transforms.ToTensor())

        # Normalization
        if components.get('normalize', {}).get('enabled', True):
            transform_list.append(transforms.Normalize(
                config['dataset']['mean'],
                config['dataset']['std']
            ))

        return transforms.Compose(transform_list)
    def process(self):
        """Main processing method with input validation"""
        # First check if input is a directory with train/test structure
        if os.path.isdir(self.datafile):
            train_dir = os.path.join(self.datafile, "train")
            test_dir = os.path.join(self.datafile, "test")
            if os.path.exists(train_dir) and os.path.exists(test_dir):
                # Verify that the folders contain images
                train_has_images = self._directory_has_images(train_dir)
                test_has_images = self._directory_has_images(test_dir)

                if train_has_images and test_has_images:
                    logger.info("Valid dataset folder structure detected. Using existing data.")
                    return train_dir, test_dir
                else:
                    logger.warning("Directory structure exists but no valid images found")

        # If not a valid folder structure, proceed with normal processing
        if self.datatype == "torchvision":
            return self._process_torchvision()
        elif self.datatype == "custom":
            return self._process_custom()
        else:
            raise ValueError("Unsupported datatype. Use 'torchvision' or 'custom'.")

    def _directory_has_images(self, directory):
        """Check if directory contains images, including in subdirectories"""
        for root, _, files in os.walk(directory):
            if any(f.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS) for f in files):
                return True
        return False


    def process_with_config(self):
        """Process dataset and return both directories and configuration"""
        # Process dataset first
        train_dir, test_dir = self.process()

        # Get base folder path (parent directory of train/test)
        base_folder = os.path.dirname(train_dir)

        # Generate default configuration using base folder
        config = self.generate_default_config(base_folder)

        return train_dir, test_dir, config

    def generate_default_config(self, folder_path: str) -> Dict:
        """Generate default configuration based on dataset properties"""
        train_folder = os.path.join(folder_path, 'train')
        test_folder = os.path.join(folder_path, 'test')

        # Get image properties from first image in training set
        first_image_path = None
        for root, _, files in os.walk(train_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    first_image_path = os.path.join(root, file)
                    break
            if first_image_path:
                break

        if not first_image_path:
            raise ValueError("No valid images found in the training directory")

        # Get image properties
        with Image.open(first_image_path) as img:
            most_common_size = img.size
            img_tensor = transforms.ToTensor()(img)
            in_channels = img_tensor.shape[0]

        # Count number of classes
        num_classes = len([d for d in os.listdir(train_folder)
                          if os.path.isdir(os.path.join(train_folder, d))])

        # Set appropriate mean and std for normalization
        mean = [0.485, 0.456, 0.406] if in_channels == 3 else [0.5]
        std = [0.229, 0.224, 0.225] if in_channels == 3 else [0.5]

        # Get dataset name from folder path
        dataset_name = os.path.basename(os.path.abspath(folder_path))

        config = {
            "dataset": {
                "name": dataset_name,
                "type": "custom",
                "in_channels": in_channels,
                "num_classes": num_classes,
                "input_size": list(most_common_size),
                "mean": mean,
                "std": std
            },
            "model": {
                "architecture": "CNN",
                "feature_dims": 128,
                "learning_rate": 0.001,
                "optimizer": {
                    "type": "Adam",
                    "weight_decay": 1e-4,
                    "momentum": 0.9
                },
                "scheduler": {
                    "type": "StepLR",
                    "step_size": 7,
                    "gamma": 0.1
                }
            },
            "training": {
                "batch_size": 32,
                "epochs": 20,
                "num_workers": min(4, os.cpu_count() or 1),
                "merge_train_test": False,
                "validation_split": 0.2,
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                },
                "checkpointing": {
                    "save_best_only": True,
                    "checkpoint_dir": "Model/cnn_checkpoints"
                }
            },
            "augmentation": {
                "enabled": True,
                "components": {
                    "normalize": {
                        "enabled": True,
                        "mean": mean,
                        "std": std
                    },
                    "resize": {
                        "enabled": True,
                        "size": list(most_common_size)
                    },
                    "horizontal_flip": {
                        "enabled": True,
                        "probability": 0.5
                    },
                    "vertical_flip": {
                        "enabled": False
                    },
                    "random_rotation": {
                        "enabled": True,
                        "degrees": 15
                    },
                    "random_crop": {
                        "enabled": True,
                        "size": list(most_common_size),
                        "padding": 4
                    },
                    "color_jitter": {
                        "enabled": True,
                        "brightness": 0.2,
                        "contrast": 0.2,
                        "saturation": 0.2,
                        "hue": 0.1
                    }
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
            }
        }

        return config

    def _process_custom(self):
        """Process custom dataset from directory or compressed file"""
        # If it's a directory, check for train/test structure
        if os.path.isdir(self.datafile):
            train_dir = os.path.join(self.datafile, "train")
            test_dir = os.path.join(self.datafile, "test")
            if os.path.exists(train_dir) and os.path.exists(test_dir):
                logger.info(f"Using existing directory structure in {self.datafile}")
                return train_dir, test_dir
            else:
                raise ValueError(f"Directory {self.datafile} must contain 'train' and 'test' subdirectories")

        # If it's a file, verify and process compressed data
        if not os.path.isfile(self.datafile):
            raise ValueError(f"Input {self.datafile} is neither a valid directory nor a file")

        # Check if it's a supported compression format
        if not any(self.datafile.lower().endswith(ext) for ext in self.SUPPORTED_COMPRESSION_FORMATS):
            raise ValueError(f"Unsupported file format. Supported formats: {', '.join(self.SUPPORTED_COMPRESSION_FORMATS)}")

        # Extract compressed data
        extract_dir = os.path.join(self.output_dir, os.path.splitext(os.path.basename(self.datafile))[0])
        os.makedirs(extract_dir, exist_ok=True)

        try:
            if self.datafile.endswith('.zip'):
                with zipfile.ZipFile(self.datafile, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            else:  # tar formats
                with tarfile.open(self.datafile, 'r:*') as tar_ref:
                    def is_within_directory(directory, target):
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        return prefix == abs_directory

                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted path traversal in tar file")
                        tar.extractall(path, members, numeric_owner=numeric_owner)

                    safe_extract(tar_ref, extract_dir)

            logger.info(f"Successfully extracted {self.datafile} to {extract_dir}")
        except Exception as e:
            raise ValueError(f"Failed to extract {self.datafile}: {str(e)}")

        # Verify extracted structure
        train_dir = os.path.join(extract_dir, "train")
        test_dir = os.path.join(extract_dir, "test")

        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            raise ValueError("Extracted dataset must have 'train' and 'test' folders")

        # Verify that extracted folders contain images
        if not self._directory_has_images(train_dir) or not self._directory_has_images(test_dir):
            raise ValueError("Extracted train/test folders must contain image files")

        return train_dir, test_dir


    def _load_existing_data(self):
        """Load existing dataset files"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return config['dataset']['train_dir'], config['dataset']['test_dir']


    def _process_torchvision(self):
        """Process torchvision datasets with progress tracking"""
        dataset_class = getattr(datasets, self.datafile, None)
        if dataset_class is None:
            raise ValueError(f"Torchvision dataset {self.datafile} not found.")

        logger.info(f"Processing {self.datafile} dataset from torchvision")

        # Download and load datasets
        train_dataset = dataset_class(root=self.output_dir, train=True, download=True)
        test_dataset = dataset_class(root=self.output_dir, train=False, download=True)

        # Setup directories
        train_dir = os.path.join(self.output_dir, self.datafile, "train")
        test_dir = os.path.join(self.output_dir, self.datafile, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Process training data with progress bar
        logger.info("Processing training dataset...")
        for idx, (img, label) in enumerate(tqdm(train_dataset, desc="Processing training data")):
            class_dir = os.path.join(train_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            img.save(os.path.join(class_dir, f"{len(os.listdir(class_dir))}.png"))

        # Process test data with progress bar
        logger.info("Processing test dataset...")
        for idx, (img, label) in enumerate(tqdm(test_dataset, desc="Processing test data")):
            class_dir = os.path.join(test_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            img.save(os.path.join(class_dir, f"{len(os.listdir(class_dir))}.png"))

        return train_dir, test_dir


    def generate_json(self, train_dir, test_dir):
        """Generate configuration JSON file based on dataset properties"""
        if os.path.isdir(self.datafile):
            dataset_name = os.path.basename(os.path.abspath(self.datafile))
        else:
            dataset_name = os.path.basename(self.datafile)

        # Check if JSON already exists
        json_path = os.path.join(self.output_dir, f"{dataset_name}.json")
        if os.path.exists(json_path):
            logger.info(f"Using existing JSON configuration from: {os.path.abspath(json_path)}")
            try:
                with open(json_path, 'r') as f:
                    existing_config = json.load(f)
                return json_path, existing_config
            except Exception as e:
                logger.error(f"Error reading existing JSON configuration: {str(e)}")
                raise

        # Only proceed with generation if no JSON exists
        logger.info("No existing JSON configuration found. Creating new one...")

        first_image_path = None
        for root, _, files in os.walk(train_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    first_image_path = os.path.join(root, file)
                    break
            if first_image_path:
                break

        if not first_image_path:
            raise ValueError("No images found in the train directory.")

        with Image.open(first_image_path) as img:
            most_common_size = img.size
            img_tensor = transforms.ToTensor()(img)
            in_channels = img_tensor.shape[0]

        num_classes = len([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])

        mean = [0.485, 0.456, 0.406] if in_channels == 3 else [0.5]
        std = [0.229, 0.224, 0.225] if in_channels == 3 else [0.5]

        json_data = {
            "dataset": {
                "name": dataset_name,
                "type": self.datatype,
                "in_channels": in_channels,
                "num_classes": num_classes,
                "input_size": list(most_common_size),
                "mean": mean,
                "std": std,
                "train_dir": train_dir,
                "test_dir": test_dir
            },
            "model": {
                "architecture": "CNN",
                "feature_dims": 128,
                "learning_rate": 0.001,
                "optimizer": {
                    "type": "Adam",
                    "weight_decay": 1e-4,
                    "momentum": 0.9
                },
                "scheduler": {
                    "type": "StepLR",
                    "step_size": 7,
                    "gamma": 0.1
                }
            },
            "training": {
                "batch_size": 32,
                "epochs": 20,
                "num_workers": 4,
                "merge_train_test": False,
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                },
                "cnn_training": {
                    "resume": True,
                    "fresh_start": False,
                    "min_loss_threshold": 0.01,
                    "checkpoint_dir": "Model/cnn_checkpoints",
                    "save_best_only": True,
                    "validation_split": 0.2
                }
            },
            "execution_flags": {
                "mode": "train_and_predict",
                "use_previous_model": True,
                "fresh_start": False,
                "use_gpu": True,
                "mixed_precision": True,
                "distributed_training": False,
                "debug_mode": False
            }
        }

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Save new JSON file
        try:
            with open(json_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
            logger.info(f"Successfully created new JSON configuration at: {os.path.abspath(json_path)}")

        except Exception as e:
            logger.error(f"Error saving JSON configuration: {str(e)}")
            raise

        return json_path, json_data


class CustomImageDataset(Dataset):
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
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
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

class FeatureExtractorCNN(nn.Module):
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

class CNNFeatureExtractor:
    def __init__(self, config: Dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.config = config
        self.device = device
        self.feature_dims = config['model']['feature_dims']
        self.learning_rate = config['model']['learning_rate']
        self.target_accuracy = config['training'].get('target_accuracy', 0.95)
        self.max_epochs = config['training'].get('max_epochs', 100)

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractorCNN(
            in_channels=config['dataset']['in_channels'],
            feature_dims=self.feature_dims
        ).to(device)

        # Configure optimizer
        optimizer_config = config['model']['optimizer']
        optimizer_params = {
            'lr': self.learning_rate,
            'weight_decay': optimizer_config.get('weight_decay', 0)
        }
        if optimizer_config['type'] == 'SGD' and 'momentum' in optimizer_config:
            optimizer_params['momentum'] = optimizer_config['momentum']

        optimizer_class = getattr(optim, optimizer_config['type'])
        self.optimizer = optimizer_class(self.feature_extractor.parameters(), **optimizer_params)

        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.history = defaultdict(list)
        self.training_log = []

        # Setup logging
        self.log_dir = os.path.join('Traininglog', config['dataset']['name'])
        os.makedirs(self.log_dir, exist_ok=True)


    def train_feature_extractor(self, train_loader: DataLoader,
                              val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_accuracy = 0.0
        patience = self.config['training'].get('patience', 5)
        patience_counter = 0

        for epoch in range(self.max_epochs):
            train_loss, train_acc = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if val_loader:
                val_loss, val_acc = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # Check if accuracy target is met
                if val_acc > self.target_accuracy:
                    logger.info(f"Reached target accuracy of {self.target_accuracy}")
                    break

                # Early stopping logic
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    patience_counter = 0
                    self._save_checkpoint('best_model.pth')
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Log progress
            logger.info(f"Epoch {epoch + 1}/{self.max_epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        return history

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.feature_extractor.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.feature_extractor(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return total_loss / len(train_loader), 100. * correct / total

    def _get_triplet_samples(self, features: torch.Tensor, labels: torch.Tensor):
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        neg_mask = ~pos_mask
        pos_mask.fill_diagonal_(False)

        valid_triplets = []
        for i in range(len(features)):
            pos_indices = torch.where(pos_mask[i])[0]
            neg_indices = torch.where(neg_mask[i])[0]

            if len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_idx = pos_indices[torch.randint(len(pos_indices), (1,))]
                neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))]
                valid_triplets.append((i, pos_idx.item(), neg_idx.item()))

        if not valid_triplets:
            return None, None, None

        indices = torch.tensor(valid_triplets, device=self.device)
        return (features[indices[:, 0]],
                features[indices[:, 1]],
                features[indices[:, 2]])

    def _validate_feature_extractor(self, val_loader: DataLoader,
                                  criterion: nn.Module) -> Tuple[float, float]:
        self.feature_extractor.eval()
        val_loss = 0.0
        valid_batches = 0
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                features = self.feature_extractor(images)

                anchors, positives, negatives = self._get_triplet_samples(features, labels)

                if anchors is not None:
                    loss = criterion(anchors, positives, negatives)
                    val_loss += loss.item()
                    valid_batches += 1

                    pos_sim = F.cosine_similarity(anchors, positives)
                    neg_sim = F.cosine_similarity(anchors, negatives)
                    correct_predictions += torch.sum(pos_sim > neg_sim).item()
                    total_samples += anchors.size(0)

        avg_loss = val_loss / valid_batches if valid_batches > 0 else float('inf')
        accuracy = (correct_predictions / total_samples * 100) if total_samples > 0 else 0
        return avg_loss, accuracy

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(images)
        return features

    def extract_dataset_features(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        all_features = []
        all_labels = []

        self.feature_extractor.eval()
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Extracting features"):
                images = images.to(self.device)
                features = self.extract_features(images)
                all_features.append(features.cpu())
                all_labels.append(labels)

        return torch.cat(all_features), torch.cat(all_labels)

    def save_features(self, features: torch.Tensor, labels: torch.Tensor, output_path: str):
        """Save extracted features in format compatible with adaptive_dbnn"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Create feature dictionary
            feature_dict = {f'feature_{i}': features[:, i].numpy()
                           for i in range(features.shape[1])}
            feature_dict['target'] = labels.numpy()

            # Convert to DataFrame and save
            df = pd.DataFrame(feature_dict)
            df.to_csv(output_path, index=False)

            logger.info(f"Saved features to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving features to {output_path}: {str(e)}")
            raise



    def save_features_old(self, features: torch.Tensor, labels: torch.Tensor, output_path: str):
        """Save extracted features in format compatible with adaptive_dbnn"""
        feature_dict = {f'feature_{i}': features[:, i].numpy() for i in range(features.shape[1])}
        feature_dict['target'] = labels.numpy()

        df = pd.DataFrame(feature_dict)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")

    def _save_checkpoint(self, checkpoint_dir: str, is_best: bool = False):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'feature_extractor': self.feature_extractor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'history': dict(self.history),
            'config': self.config
        }

        filename = f"{self.config['dataset']['name']}_{'best' if is_best else 'checkpoint'}.pth"
        path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        if not os.path.exists(checkpoint_path):
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.history = defaultdict(list, checkpoint['history'])
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        return True

    def log_training_metrics(self, epoch: int, train_loss: float, train_acc: float,
                           test_loss: Optional[float] = None, test_acc: Optional[float] = None,
                           train_loader: DataLoader = None, test_loader: Optional[DataLoader] = None):
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'train_samples': len(train_loader.dataset) if train_loader else None,
            'test_samples': len(test_loader.dataset) if test_loader else None,
        }
        self.training_log.append(metrics)

        log_df = pd.DataFrame(self.training_log)
        log_path = os.path.join(self.log_dir, f'training_metrics.csv')
        log_df.to_csv(log_path, index=False)

        logger.info(f"Epoch {epoch}: "
                   f"Train [{metrics['train_samples']} samples] Loss {train_loss:.4f}, Acc {train_acc:.2f}%"
                   + (f", Test [{metrics['test_samples']} samples] Loss {test_loss:.4f}, "
                      f"Acc {test_acc:.2f}%" if test_loss is not None else ""))

class CombinedDataset(Dataset):
    def __init__(self, train_dataset, test_dataset):
        self.combined_data = ConcatDataset([train_dataset, test_dataset])

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        return self.combined_data[idx]

def setup_logging(log_dir='logs'):
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


class ConfigManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def generate_merged_configs(self, dataset_name: str, config: Dict, feature_dims: int) -> Tuple[str, str]:
        """Generate configurations for merged dataset without train/test references"""
        # Generate DBNN configuration
        dbnn_config = self.generate_dbnn_config(config)

        # Modify file paths to remove train/test references
        dbnn_config['file_path'] = f"{dataset_name}.csv"

        # Generate data configuration
        data_config = self.generate_data_conf(config, feature_dims)
        data_config['file_path'] = f"{dataset_name}.csv"

        # Save configurations
        configs_dir = 'data'
        os.makedirs(configs_dir, exist_ok=True)

        # Save DBNN config
        dbnn_config_path = os.path.join(configs_dir, f"adaptive_dbnn.conf")
        with open(dbnn_config_path, 'w') as f:
            json.dump(dbnn_config, f, indent=4)

        # Save data config
        data_config_path = os.path.join(configs_dir, f"{dataset_name}.conf")
        with open(data_config_path, 'w') as f:
            json.dump(data_config, f, indent=4)

        return dbnn_config_path, data_config_path

    def update_config_for_merged(self, config: Dict) -> Dict:
        """Update main configuration to reflect merged dataset"""
        # Create a copy to avoid modifying the original
        updated_config = copy.deepcopy(config)

        # Remove train/test specific paths if they exist
        if 'dataset' in updated_config:
            updated_config['dataset'].pop('train_dir', None)
            updated_config['dataset'].pop('test_dir', None)

        # Update training settings
        if 'training' in updated_config:
            updated_config['training']['merge_train_test'] = True
            # Remove validation split since we're using all data
            updated_config['training'].pop('validation_split', None)

        return updated_config

    def edit_config_file(self, config_path: str) -> dict:
        """Edit configuration file using system's default editor"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            # Open in default text editor
            if os.name == 'nt':  # Windows
                os.system(f'notepad {config_path}')
            elif os.name == 'posix':  # Linux/Mac
                editor = os.environ.get('EDITOR', 'nano')  # Default to nano if EDITOR not set
                os.system(f'{editor} {config_path}')

            # Give the editor a moment to close and save the file
            time.sleep(1)

            # Reload and validate config after editing
            with open(config_path, 'r') as f:
                try:
                    config = json.load(f)
                    return config
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in edited config file: {str(e)}")
                    raise ValueError(f"Configuration file contains invalid JSON after editing: {str(e)}")

        except Exception as e:
            logger.error(f"Error editing configuration file {config_path}: {str(e)}")
            raise

    def manage_configuration(self, config_path: str, config_type: str) -> dict:
        """Manage configuration file with user interaction"""
        logger.info(f"Managing {config_type} configuration...")

        if os.path.exists(config_path):
            edit = input(f"Would you like to edit the {config_type} configuration? (y/n): ").lower() == 'y'
            if edit:
                config = self.edit_config_file(config_path)
                logger.info(f"{config_type} configuration updated successfully")
                return config
            else:
                with open(config_path, 'r') as f:
                    return json.load(f)
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            return None

    def manage_all_configs(self, dataset_name: str, json_config: dict):
        """Manage all configuration files for the dataset"""
        # Main JSON config
        json_path = os.path.join(self.base_dir, f"{dataset_name}.json")
        if not os.path.exists(json_path):
            with open(json_path, 'w') as f:
                json.dump(json_config, f, indent=4)
        updated_json = self.manage_configuration(json_path, "main")

        # DBNN config
        dbnn_config_path = os.path.join(self.base_dir, f"adaptive_dbnn.conf")
        if not os.path.exists(dbnn_config_path):
            dbnn_config = self.generate_dbnn_config(updated_json)
            with open(dbnn_config_path, 'w') as f:
                json.dump(dbnn_config, f, indent=4)
        updated_dbnn = self.manage_configuration(dbnn_config_path, "DBNN")

        # Data config
        data_config_path = os.path.join(self.base_dir, f"{dataset_name}.conf")
        if not os.path.exists(data_config_path):
            data_config = self.generate_data_config(updated_json)
            with open(data_config_path, 'w') as f:
                json.dump(data_config, f, indent=4)
        updated_data = self.manage_configuration(data_config_path, "data")

        return {
            'main': updated_json if updated_json else json_config,
            'dbnn': updated_dbnn,
            'data': updated_data
        }

    def generate_dbnn_config(self, json_config: dict) -> dict:
        """Generate DBNN configuration from main config"""
        dataset_name = json_config['dataset']['name']
        feature_dims = json_config['model']['feature_dims']

        return {
            "file_path": f"{dataset_name}.csv",
            "column_names": [f"feature_{i}" for i in range(feature_dims)] + ["target"],
            "separator": ",",
            "has_header": True,
            "target_column": "target",
            "likelihood_config": {
                "feature_group_size": 2,
                "max_combinations": min(1000, feature_dims * (feature_dims - 1) // 2),
                "bin_sizes": [20]
            },
            "active_learning": {
                "tolerance": 1.0,
                "cardinality_threshold_percentile": 95,
                "strong_margin_threshold": 0.3,
                "marginal_margin_threshold": 0.1,
                "min_divergence": 0.1
            },
            "training_params": {
                "trials": json_config['training']['epochs'],
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "learning_rate": json_config['model']['learning_rate'],
                "random_seed": 42,
                "epochs": json_config['training']['epochs'],
                "test_fraction": json_config['training'].get('validation_split', 0.2),
                "enable_adaptive": True,
                "modelType": "Histogram",
                "compute_device": "cuda" if json_config['execution_flags']['use_gpu'] else "cpu",
                "Save_training_epochs": True,
                "training_save_path": f"training_data/{dataset_name}"
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

    def generate_data_config(self, json_config: dict) -> dict:
        """Generate data configuration from main config"""
        return {
            "dataset": {
                "name": json_config['dataset']['name'],
                "type": json_config['dataset']['type'],
                "input_size": json_config['dataset']['input_size'],
                "num_classes": json_config['dataset']['num_classes']
            },
            "preprocessing": {
                "normalize": True,
                "mean": json_config['dataset']['mean'],
                "std": json_config['dataset']['std']
            },
            "augmentation": json_config.get('augmentation', {})
        }

    def load_or_generate_config(self) -> dict:
        """Load existing configuration or generate new one"""
        if self.check_existing_files():
            config_path = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(self.datafile))[0]}.json")
            if os.path.exists(config_path):
                logger.info(f"Loading existing configuration from {config_path}")
                with open(config_path, 'r') as f:
                    return json.load(f)

        # Process dataset and generate config
        train_dir, test_dir = self.process()
        return self.generate_default_config(train_dir, test_dir)

    def process_with_config(self):
        """Process dataset and return both directories and configuration"""
        config = self.load_or_generate_config()
        return config['dataset']['train_dir'], config['dataset']['test_dir'], config

    def generate_dbnn_conf(self, json_config: dict, dataset_name: str) -> dict:
        """Generate DBNN configuration based on JSON config"""
        training_params = json_config.get('training', {})
        model_params = json_config.get('model', {})

        return {
            "training_params": {
                "trials": training_params.get('epochs', 100),
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "learning_rate": model_params.get('learning_rate', 0.001),
                "random_seed": 42,
                "epochs": training_params.get('epochs', 1000),
                "test_fraction": training_params.get('validation_split', 0.2),
                "enable_adaptive": True,
                "modelType": "Histogram",
                "compute_device": "cuda" if json_config['execution_flags'].get('use_gpu', True) else "cpu",
                "use_interactive_kbd": False,
                "debug_enabled": json_config['execution_flags'].get('debug_mode', True),
                "Save_training_epochs": True,
                "training_save_path": f"training_data/{dataset_name}"
            },
            "execution_flags": {
                "train": True,
                "train_only": False,
                "predict": True,
                "gen_samples": False,
                "fresh_start": json_config['execution_flags'].get('fresh_start', False),
                "use_previous_model": json_config['execution_flags'].get('use_previous_model', True)
            }
        }

    def generate_data_conf(self, json_config: dict, feature_dims: int) -> dict:
        """Generate data configuration based on JSON config"""
        dataset_config = json_config.get('dataset', {})

        return {
            "file_path": f"{dataset_config.get('name', 'dataset')}.csv",
            "column_names": [f"feature_{i}" for i in range(feature_dims)] + ["target"],
            "separator": ",",
            "has_header": True,
            "target_column": "target",
            "likelihood_config": {
                "feature_group_size": 2,
                "max_combinations": min(1000, feature_dims * (feature_dims - 1) // 2),
                "bin_sizes": [20]
            },
            "active_learning": {
                "tolerance": 1.0,
                "cardinality_threshold_percentile": 95,
                "strong_margin_threshold": 0.3,
                "marginal_margin_threshold": 0.1,
                "min_divergence": 0.1
            },
            "training_params": {
                "Save_training_epochs": True,
                "training_save_path": f"training_data/{dataset_config.get('name', 'dataset')}"
            },
            "modelType": "Histogram"
        }

    def edit_conf_file(self, conf_data: dict, filepath: str) -> dict:
        """Edit configuration file with user input"""
        def edit_nested_dict(d, parent_key=''):
            for key, value in d.items():
                if isinstance(value, dict):
                    print(f"\nEditing section: {key}")
                    edit_nested_dict(value, f"{parent_key}{key}.")
                else:
                    current_value = str(value)
                    new_value = input(f"{parent_key}{key} [{current_value}]: ").strip()
                    if new_value:
                        # Convert value to appropriate type
                        if isinstance(value, bool):
                            d[key] = new_value.lower() in ('true', 't', 'yes', 'y', '1')
                        elif isinstance(value, int):
                            d[key] = int(new_value)
                        elif isinstance(value, float):
                            d[key] = float(new_value)
                        else:
                            d[key] = new_value

        print(f"\nEditing configuration file: {filepath}")
        print("Press Enter to keep current value, or enter new value to change")

        edit_nested_dict(conf_data)

        # Save the updated configuration
        with open(filepath, 'w') as f:
            json.dump(conf_data, f, indent=4)

        return conf_data

    def edit_csv_file(self, csv_path: str) -> None:
        """Edit CSV file with user input"""
        try:
            df = pd.read_csv(csv_path)
            print(f"\nEditing CSV file: {csv_path}")
            print("\nCurrent column names:", df.columns.tolist())

            # Allow column renaming
            rename_cols = input("\nWould you like to rename any columns? (y/n): ").lower() == 'y'
            if rename_cols:
                while True:
                    old_name = input("\nEnter column name to rename (or press Enter to finish): ").strip()
                    if not old_name:
                        break
                    if old_name in df.columns:
                        new_name = input(f"Enter new name for {old_name}: ").strip()
                        if new_name:
                            df = df.rename(columns={old_name: new_name})
                    else:
                        print(f"Column {old_name} not found")

            # Allow basic data manipulation
            while True:
                print("\nData manipulation options:")
                print("1. View sample data")
                print("2. Filter rows")
                print("3. Handle missing values")
                print("4. Save and exit")

                choice = input("\nEnter your choice (1-4): ").strip()

                if choice == '1':
                    print("\nSample data:")
                    print(df.head())
                    print("\nDataset info:")
                    print(df.info())

                elif choice == '2':
                    column = input("Enter column name to filter: ").strip()
                    if column in df.columns:
                        condition = input(f"Enter condition (e.g., '> 0' or '== \"value\"'): ").strip()
                        try:
                            df = df.query(f"{column} {condition}")
                            print(f"Dataset filtered. New shape: {df.shape}")
                        except Exception as e:
                            print(f"Error applying filter: {str(e)}")

                elif choice == '3':
                    print("\nMissing value handling:")
                    print(df.isnull().sum())
                    column = input("Enter column name to handle missing values: ").strip()
                    if column in df.columns:
                        strategy = input("Choose strategy (drop/mean/median/mode/value): ").strip()
                        if strategy == 'drop':
                            df = df.dropna(subset=[column])
                        elif strategy in ['mean', 'median', 'mode']:
                            fill_value = getattr(df[column], strategy)()
                            df[column] = df[column].fillna(fill_value)
                        else:
                            df[column] = df[column].fillna(strategy)

                elif choice == '4':
                    df.to_csv(csv_path, index=False)
                    print(f"Changes saved to {csv_path}")
                    break

        except Exception as e:
            print(f"Error editing CSV file: {str(e)}")

    def verify_config(self, config: dict, config_type: str = "main") -> bool:
        """Verify configuration has all required fields based on config type"""
        if config_type == "main":
            required_fields = {
                'dataset': ['name', 'type', 'in_channels', 'num_classes', 'input_size', 'mean', 'std'],
                'model': ['feature_dims', 'learning_rate'],
                'training': ['batch_size', 'epochs', 'num_workers'],
                'execution_flags': ['mode', 'use_gpu']
            }
        elif config_type == "dbnn":
            required_fields = {
                'file_path': None,
                'column_names': None,
                'target_column': None,
                'training_params': ['trials', 'epochs', 'learning_rate'],
                'execution_flags': ['train', 'predict']
            }

        for section, fields in required_fields.items():
            if section not in config:
                print(f"Missing section: {section}")
                return False
            if fields:  # If fields is not None, verify them
                for field in fields:
                    if field not in config[section]:
                        print(f"Missing field: {section}.{field}")
                        return False
        return True

    def handle_config_editing(self, config_path: str, config_type: str = "main") -> dict:
        """Handle configuration editing and validation"""
        edit = input(f"Would you like to edit the {config_type} configuration? (y/n): ").lower() == 'y'
        if edit:
            self.edit_config_file(config_path)
            with open(config_path, 'r') as f:
                config = json.load(f)
                if not self.verify_config(config, config_type):
                    raise ValueError(f"Configuration still invalid after editing: {config_path}")
                return config
        return None



def plot_training_history(history: Dict, save_path: Optional[str] = None):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Feature Extractor Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.close()

def get_dataset(config: Dict, transform, use_cpu: bool = False):
    """
    Get dataset based on configuration. For custom datasets, uses folder/train and folder/test structure.
    """
    dataset_config = config['dataset']
    merge_datasets = config.get('training', {}).get('merge_train_test', False)

    if dataset_config['type'] == 'torchvision':
        train_dataset = getattr(torchvision.datasets, dataset_config['name'].upper())(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = getattr(torchvision.datasets, dataset_config['name'].upper())(
            root='./data', train=False, download=True, transform=transform
        )
    elif dataset_config['type'] == 'custom':
        # For custom datasets, use the base folder path
        base_folder = os.path.join("data",dataset_config['name'])
        train_folder = os.path.join(base_folder, 'train')
        test_folder = os.path.join(base_folder, 'test')

        train_dataset = CustomImageDataset(
            data_dir=train_folder,
            transform=transform
        )
        test_dataset = CustomImageDataset(
            data_dir=test_folder,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config['type']}")

    if merge_datasets:
        return CombinedDataset(train_dataset, test_dataset), None
    return train_dataset, test_dataset

class TrainingProgress:
    def __init__(self):
        self.train_loss = 0.0
        self.train_acc = 0.0
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.best_val_acc = 0.0
        self.epoch_start_time = None

    def update_metrics(self, train_loss, train_acc, val_loss=None, val_acc=None):
        self.train_loss = train_loss
        self.train_acc = train_acc
        if val_loss is not None:
            self.val_loss = val_loss
        if val_acc is not None:
            self.val_acc = val_acc
            self.best_val_acc = max(self.best_val_acc, val_acc)

    def get_progress_description(self):
        desc = f'Loss: {self.train_loss:.4f}, Acc: {self.train_acc:.2f}%'
        if hasattr(self, 'val_loss'):
            desc += f' | Val Loss: {self.val_loss:.4f}, Val Acc: {self.val_acc:.2f}%'
        return desc

class CNNTrainer:
    def __init__(self, config: Dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.config = config
        self.device = device
        self.feature_dims = config['model']['feature_dims']
        self.learning_rate = config['model']['learning_rate']
        self.progress = TrainingProgress()

        # Initialize model and check for previous checkpoint
        self.feature_extractor = self._initialize_model()

        # Configure optimizer after model initialization
        optimizer_config = config['model']['optimizer']
        optimizer_params = {
            'lr': self.learning_rate,
            'weight_decay': optimizer_config.get('weight_decay', 0)
        }
        if optimizer_config['type'] == 'SGD' and 'momentum' in optimizer_config:
            optimizer_params['momentum'] = optimizer_config['momentum']

        optimizer_class = getattr(optim, optimizer_config['type'])
        self.optimizer = optimizer_class(self.feature_extractor.parameters(), **optimizer_params)

        # Training metrics
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.history = defaultdict(list)
        self.training_log = []
        self.training_start_time = None

        # Setup logging directory
        self.log_dir = os.path.join('Traininglog', config['dataset']['name'])
        os.makedirs(self.log_dir, exist_ok=True)


    def save_features(self, features: torch.Tensor, labels: torch.Tensor, output_path: str):
        """Save extracted features to CSV"""
        feature_dict = {f'feature_{i}': features[:, i].numpy() for i in range(features.shape[1])}
        feature_dict['target'] = labels.numpy()

        df = pd.DataFrame(feature_dict)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")

    def save_merged_features(self, features: torch.Tensor, labels: torch.Tensor,
                           dataset_name: str, output_dir: str) -> str:
        """Save features for merged dataset without train/test suffixes"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{dataset_name}.csv")
        self.save_features(features, labels, output_path)
        return output_path


    def _initialize_model(self) -> nn.Module:
        """Initialize model, optionally loading from previous checkpoint
        Returns:
            nn.Module: Initialized model, either fresh or loaded from checkpoint
        """
        # Create new model instance
        model = FeatureExtractorCNN(
            in_channels=self.config['dataset']['in_channels'],
            feature_dims=self.feature_dims
        ).to(self.device)

        # Check if we should use previous model (default to True unless explicitly set False)
        if not self.config['execution_flags'].get('fresh_start', False):
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path:
                try:
                    logger.info(f"Found previous checkpoint at {checkpoint_path}")
                    self._load_checkpoint(checkpoint_path, model)
                    logger.info("Successfully initialized model from previous checkpoint")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {str(e)}. Starting fresh.")

        logger.info("Initializing fresh model")
        return model

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint for the current dataset
        Returns:
            Optional[str]: Path to latest checkpoint or None if not found
        """
        dataset_name = self.config['dataset']['name']
        checkpoint_dir = os.path.join("Model", "cnn_checkpoints")

        if not os.path.exists(checkpoint_dir):
            logger.info(f"No checkpoint directory found at {checkpoint_dir}")
            return None

        # Look for both best and latest checkpoints
        checkpoints = []

        # Check for best model first
        best_model_path = os.path.join(checkpoint_dir, f"{dataset_name}_best.pth")
        if os.path.exists(best_model_path):
            checkpoints.append(best_model_path)

        # Then check for checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoints.append(checkpoint_path)

        # Finally, look for any other checkpoints matching the dataset name
        pattern = os.path.join(checkpoint_dir, f"{dataset_name}*.pth")
        checkpoints.extend(glob.glob(pattern))

        if not checkpoints:
            logger.info(f"No checkpoints found for {dataset_name}")
            return None

        # Sort checkpoints by modification time and get the latest
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        logger.info(f"Latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint

    def _load_checkpoint(self, checkpoint_path: str, model: nn.Module) -> None:
        """Load model and training state from checkpoint
        Args:
            checkpoint_path (str): Path to checkpoint file
            model (nn.Module): Model to load state into
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            logger.info("Checkpoint loaded successfully")

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Load model state
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    logger.warning("No model state dict found in checkpoint")
                    return

                # Load training state if available
                self.current_epoch = checkpoint.get('epoch', 0)
                self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))

                # Load optimizer state if available and optimizer exists
                if 'optimizer_state_dict' in checkpoint and self.optimizer:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("Optimizer state loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {str(e)}")

                # Load training history if available
                if 'history' in checkpoint:
                    self.history = defaultdict(list, checkpoint['history'])
                    logger.info("Training history loaded")
            else:
                # Treat checkpoint as just the model state dict
                model.load_state_dict(checkpoint)
                logger.info("Loaded model state only")

        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint"""
        dataset_name = self.config['dataset']['name']
        checkpoint_dir = os.path.join("Model", "cnn_checkpoints", dataset_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Prepare checkpoint data
        checkpoint = {
            'epoch': self.current_epoch,
            'state_dict': self.feature_extractor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'history': dict(self.history),
            'config': self.config
        }

        # Save checkpoint
        filename = f"{dataset_name}_{'best' if is_best else 'latest'}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)

        # Save timestamp file for tracking
        timestamp_path = os.path.join(checkpoint_dir, f"{filename}.time")
        with open(timestamp_path, 'w') as f:
            f.write(str(time.time()))

        logger.info(f"Saved {'best' if is_best else 'latest'} checkpoint to {checkpoint_path}")

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """Train the model with checkpoint handling"""
        target_accuracy = self.config['training'].get('target_accuracy', 0.95)
        min_loss = self.config['training'].get('min_loss', 0.01)
        max_epochs = self.config['training'].get('epochs', 100)
        patience = self.config['training'].get('early_stopping', {}).get('patience', 5)

        patience_counter = 0
        best_val_metric = float('inf')

        logger.info(f"Starting training from epoch {self.current_epoch + 1}")
        logger.info(f"Target accuracy: {target_accuracy}, min loss: {min_loss}")

        # Initialize training timer
        self.training_start_time = time.time()

        try:
            for epoch in range(self.current_epoch, max_epochs):
                self.current_epoch = epoch

                # Training and validation steps...
                train_loss, train_acc = self._train_epoch(train_loader, epoch)

                if val_loader:
                    val_loss, val_acc = self._validate(val_loader)
                    current_metric = val_loss
                else:
                    val_loss, val_acc = None, None
                    current_metric = train_loss

                # Update progress and save checkpoints
                self.log_training_metrics(epoch, train_loss, train_acc, val_loss, val_acc)

                # Save regular checkpoint
                self._save_checkpoint(is_best=False)

                # Save best checkpoint if improved
                if current_metric < best_val_metric:
                    best_val_metric = current_metric
                    patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    patience_counter += 1

                # Check stopping conditions...
                if train_acc >= target_accuracy and train_loss <= min_loss:
                    logger.info(f"Reached targets: Accuracy {train_acc:.2f}% >= {target_accuracy}%, "
                              f"Loss {train_loss:.4f} <= {min_loss}")
                    break

                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint(is_best=False)  # Save checkpoint on interruption

        return self.history

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train one epoch with progress bar"""
        self.feature_extractor.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Create progress bar for batches
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}',
                         unit='batch', position=1, leave=False)

        for batch_idx, (inputs, targets) in enumerate(batch_pbar):
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

            # Update batch progress bar
            batch_loss = running_loss / (batch_idx + 1)
            batch_acc = 100. * correct / total
            batch_pbar.set_description(
                f'Epoch {epoch+1} | Loss: {batch_loss:.4f}, Acc: {batch_acc:.2f}%'
            )

        batch_pbar.close()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model with progress bar"""
        self.feature_extractor.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        val_pbar = tqdm(val_loader, desc='Validation',
                       unit='batch', position=1, leave=False)

        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.feature_extractor(inputs)
                loss = F.cross_entropy(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update validation progress bar
                val_loss = running_loss / (total / targets.size(0))
                val_acc = 100. * correct / total
                val_pbar.set_description(
                    f'Validation | Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%'
                )

        val_pbar.close()
        return running_loss / len(val_loader), 100. * correct / total

    def log_training_metrics(self, epoch: int, train_loss: float, train_acc: float,
                         test_loss: Optional[float] = None, test_acc: Optional[float] = None,
                         train_loader: DataLoader = None, test_loader: Optional[DataLoader] = None):
        """Log training metrics and save to file"""
        if self.training_start_time is None:
            self.training_start_time = time.time()

        elapsed_time = time.time() - self.training_start_time

        metrics = {
            'epoch': epoch,
            'elapsed_time': elapsed_time,
            'elapsed_time_formatted': str(timedelta(seconds=int(elapsed_time))),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }

        self.training_log.append(metrics)

        # Save to CSV
        log_df = pd.DataFrame(self.training_log)
        log_path = os.path.join(self.log_dir, 'training_metrics.csv')
        log_df.to_csv(log_path, index=False)

        # Log to console
        log_message = (f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc:.2f}%")
        if test_loss is not None:
            log_message += f", Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"

        logger.info(log_message)

    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from data loader"""
        self.feature_extractor.eval()
        features = []
        labels = []

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Extracting features"):
                inputs = inputs.to(self.device)
                outputs = self.feature_extractor(inputs)
                features.append(outputs.cpu())
                labels.append(targets)

        return torch.cat(features), torch.cat(labels)

    def save_features(self, features: torch.Tensor, labels: torch.Tensor, output_path: str):
        """Save extracted features to CSV"""
        feature_dict = {f'feature_{i}': features[:, i].numpy() for i in range(features.shape[1])}
        feature_dict['target'] = labels.numpy()

        df = pd.DataFrame(feature_dict)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join('Model', 'cnn_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'state_dict': self.feature_extractor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'history': dict(self.history),
            'config': self.config
        }

        filename = f"{self.config['dataset']['name']}_{'best' if is_best else 'checkpoint'}.pth"
        path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

def main(args=None):
    try:
        logger = setup_logging()
        logger.info("Starting CNN training process...")

        # Initialize ConfigManager
        config_manager = ConfigManager(base_dir='configs')

        # Handle configuration and dataset setup
        if args:
            if args.config:
                # Load configuration from file
                logger.info(f"Loading configuration from {args.config}")
                with open(args.config, 'r') as f:
                    config = json.load(f)
            else:
                # Use command line arguments to create configuration
                if not args.data_type:
                    raise ValueError("Data type must be specified when not using a config file")
                if not args.data:
                    raise ValueError("Data path/name must be specified when not using a config file")

                # Initialize dataset processor with CLI arguments
                processor = DatasetProcessor(
                    datafile=args.data,
                    datatype=args.data_type.lower(),
                    output_dir=args.output_dir
                )

                # Process dataset
                train_dir, test_dir = processor.process()

                # Generate configuration
                config = processor.generate_default_config(os.path.dirname(train_dir))

                # Update config with CLI arguments
                config['dataset']['type'] = args.data_type.lower()
                config['dataset']['name'] = os.path.basename(args.data) if os.path.isfile(args.data) else args.data
                if args.batch_size:
                    config['training']['batch_size'] = args.batch_size
                if args.epochs:
                    config['training']['epochs'] = args.epochs
                if args.workers:
                    config['training']['num_workers'] = args.workers
                if args.learning_rate:
                    config['model']['learning_rate'] = args.learning_rate
                if args.cpu:
                    config['execution_flags']['use_gpu'] = False
        else:
            # Interactive mode (existing implementation)
            datafile = input("Enter dataset name or path (default: MNIST): ").strip() or "MNIST"
            datatype = input("Enter dataset type (torchvision/custom) (default: torchvision): ").strip() or "torchvision"
            if datatype == "torchvision":
                datafile = datafile.upper()

            processor = DatasetProcessor(datafile=datafile, datatype=datatype)
            train_dir, test_dir = processor.process()
            config_path, config = processor.generate_json(train_dir, test_dir)
            logger.info(f"Using configuration from: {config_path}")

            merge_datasets = input("Merge train and test datasets? (y/n, default: n): ").lower() == 'y'
            if merge_datasets:
                config = config_manager.update_config_for_merged(config)

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() and
                            config['execution_flags'].get('use_gpu', True) else 'cpu')
        logger.info(f"Using device: {device}")

        # Get transforms and create datasets
        processor = DatasetProcessor(datafile=config['dataset']['name'],
                                  datatype=config['dataset']['type'])
        transform = processor.get_transforms(config)
        train_dataset, test_dataset = get_dataset(config, transform)

        # Create data loaders
        if config['training'].get('merge_train_test', False) and test_dataset:
            logger.info("Creating merged dataset loader...")
            combined_dataset = ConcatDataset([train_dataset, test_dataset])
            train_loader = DataLoader(
                combined_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=config['training']['num_workers'],
                pin_memory=device.type=='cuda'
            )
            test_loader = None
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=config['training']['num_workers'],
                pin_memory=device.type=='cuda'
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['training']['num_workers'],
                pin_memory=device.type=='cuda'
            ) if test_dataset else None

        # Initialize trainer and run training
        trainer = CNNTrainer(config=config, device=device)
        history = trainer.train(train_loader, test_loader)

        # Extract and save features
        logger.info("Extracting features...")
        output_dir = os.path.join('data', config['dataset']['name'])

        if config['training'].get('merge_train_test', False):
            logger.info("Extracting features for merged dataset...")
            features, labels = trainer.extract_features(train_loader)
            output_path = trainer.save_merged_features(features, labels, config['dataset']['name'], output_dir)

            dbnn_config_path, data_config_path = config_manager.generate_merged_configs(
                config['dataset']['name'],
                config,
                config['model']['feature_dims']
            )

            logger.info(f"Merged dataset features saved to: {output_path}")
            logger.info(f"DBNN configuration saved to: {dbnn_config_path}")
            logger.info(f"Data configuration saved to: {data_config_path}")
        else:
            train_features, train_labels = trainer.extract_features(train_loader)
            train_output_path = os.path.join(output_dir, f"{config['dataset']['name']}_train.csv")
            trainer.save_features(train_features, train_labels, train_output_path)

            if test_loader:
                test_features, test_labels = trainer.extract_features(test_loader)
                test_output_path = os.path.join(output_dir, f"{config['dataset']['name']}_test.csv")
                trainer.save_features(test_features, test_labels, test_output_path)

                logger.info(f"Training features saved to: {train_output_path}")
                logger.info(f"Test features saved to: {test_output_path}")

        logger.info("Process completed successfully!")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        if args and hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Feature Extractor Training')

    # Configuration options
    parser.add_argument('--config', type=str, help='path to configuration file')
    parser.add_argument('--data_type', type=str, choices=['torchvision', 'custom'],
                        help='type of dataset (torchvision or custom)')
    parser.add_argument('--data', type=str,
                        help='dataset name for torchvision or path for custom dataset')

    # Training parameters
    parser.add_argument('--batch_size', type=int, help='batch size for training')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--workers', type=int, help='number of data loading workers')
    parser.add_argument('--learning_rate', type=float, help='learning rate')

    # Output and execution options
    parser.add_argument('--output-dir', type=str, default='training_results',
                        help='output directory (default: training_results)')
    parser.add_argument('--cpu', action='store_true',
                        help='force CPU usage')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode')
    parser.add_argument('--merge-datasets', action='store_true',
                        help='merge train and test datasets')

    args = parser.parse_args()
    main(args)
