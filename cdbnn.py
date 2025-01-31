import torch
import copy
import sys
import gc
import os
import torch
import traceback
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
from tqdm import tqdm
import random
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import torch.multiprocessing
# Set sharing strategy at the start
torch.multiprocessing.set_sharing_strategy('file_system')
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
        self.dataset_name = os.path.basename(os.path.normpath(datafile))

        # Create dataset-specific output directory
        self.dataset_dir = os.path.join(output_dir, self.dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Set paths for dataset files
        self.config_path = os.path.join(output_dir, f"{self.dataset_name}.json")
        self.conf_path = os.path.join(output_dir, f"{self.dataset_name}.conf")
        self.csv_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.csv")

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

        # Update config with correct paths
        config['dataset'].update({
            'train_dir': train_dir,
            'test_dir': test_dir if test_dir else ''
        })

        return train_dir, test_dir, config

    def read_config_file(self, file_path: str) -> Dict:
        """
        Read configuration file and handle comments.
        Supports multiple comment styles:
        - // Single line comments
        - /* */ Multi-line comments
        - _comment field comments
        - # Python-style comments
        """
        def strip_comments(text: str) -> str:
            # State tracking for strings and comments
            in_string = False
            string_char = None
            i = 0
            result = []
            length = len(text)

            while i < length:
                char = text[i]

                # Handle strings (to avoid removing comments inside strings)
                if char in ('"', "'") and (i == 0 or text[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                    result.append(char)
                    i += 1
                    continue

                if in_string:
                    result.append(char)
                    i += 1
                    continue

                # Handle multi-line comments /* */
                if char == '/' and i + 1 < length and text[i + 1] == '*':
                    while i < length - 1 and not (text[i] == '*' and text[i + 1] == '/'):
                        i += 1
                    i += 2
                    continue

                # Handle single-line comments //
                if char == '/' and i + 1 < length and text[i + 1] == '/':
                    while i < length and text[i] != '\n':
                        i += 1
                    continue

                # Handle Python-style comments #
                if char == '#':
                    while i < length and text[i] != '\n':
                        i += 1
                    continue

                result.append(char)
                i += 1

            return ''.join(result)

        def clean_dict(d: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
            """Remove _comment fields and handle nested structures"""
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()
                       if not (k.endswith('_comment') or k.startswith('_comment'))}
            elif isinstance(d, list):
                return [clean_dict(item) for item in d]
            return d

        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()

            # First, strip C-style and Python-style comments
            cleaned_content = strip_comments(content)

            try:
                # Try to parse as JSON
                config = json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                # Log the error location
                lines = cleaned_content.split('\n')
                error_line = lines[e.lineno - 1]
                logger.error(f"JSON parsing error at line {e.lineno}, column {e.colno}")
                logger.error(f"Line content: {error_line}")
                logger.error(f"Error message: {str(e)}")
                raise ValueError(f"Invalid JSON in configuration file: {str(e)}")

            # Remove _comment fields
            config = clean_dict(config)

            # Validate required fields based on file type
            filename = os.path.basename(file_path)
            if filename == 'adaptive_dbnn.conf':
                required_fields = {'training_params', 'execution_flags'}
            elif filename.endswith('.conf'):
                required_fields = {'file_path', 'column_names', 'target_column'}
            else:  # .json files
                required_fields = {'dataset', 'model', 'training', 'execution_flags'}

            missing_fields = required_fields - set(config.keys())
            if missing_fields:
                raise ValueError(f"Missing required fields in configuration: {missing_fields}")

            return config

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading configuration file {file_path}: {str(e)}")
            raise

    def write_config_file(self, config: Dict, file_path: str, include_comments: bool = True) -> None:
        """
        Write configuration to file with proper formatting and comment handling.
        Args:
            config: Configuration dictionary
            file_path: Output file path
            include_comments: Whether to include comments in output
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Format config for output
            if include_comments:
                formatted_config = json.dumps(config, indent=4)
            else:
                # Remove comments before writing
                clean_config = self.clean_dict(config)
                formatted_config = json.dumps(clean_config, indent=4)

            with open(file_path, 'w') as f:
                f.write(formatted_config)

            logger.info(f"Configuration saved to: {file_path}")

        except Exception as e:
            logger.error(f"Error writing configuration to {file_path}: {str(e)}")
            raise

    def _detect_image_properties(self, folder_path: str) -> Tuple[Tuple[int, int], int]:
        """Detect image size and channels from actual images in the dataset"""
        img_formats = self.SUPPORTED_IMAGE_EXTENSIONS
        size_counts = defaultdict(int)
        channel_counts = defaultdict(int)

        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in img_formats):
                    try:
                        with Image.open(os.path.join(root, file)) as img:
                            # Convert to tensor to properly handle channels and dimensions
                            tensor = transforms.ToTensor()(img)
                            height, width = tensor.shape[1], tensor.shape[2]
                            channels = tensor.shape[0]

                            size_counts[(width, height)] += 1
                            channel_counts[channels] += 1
                    except Exception as e:
                        logger.warning(f"Could not read image {file}: {str(e)}")
                        continue

            # Stop after checking 50 images to balance accuracy vs performance
            if sum(size_counts.values()) >= 50:
                break

        if not size_counts:
            raise ValueError(f"No valid images found in {folder_path}")

        # Get most common dimensions and channels
        input_size = max(size_counts, key=size_counts.get)
        in_channels = max(channel_counts, key=channel_counts.get)

        return input_size, in_channels

    def generate_default_config(self, folder_path: str) -> Dict[str, Dict]:
        """
        Generate three configuration files with exact structure and comments.
        Returns dictionary containing all three configurations.
        """
        try:
            # Normalize paths and get dataset name
            folder_path = os.path.abspath(folder_path)
            dataset_name = os.path.basename(os.path.normpath(folder_path))

            # Get image properties from first image
            train_folder = os.path.join(folder_path, 'train')
            if not os.path.exists(train_folder):
                raise ValueError(f"Training directory not found: {train_folder}")

            # Detect properties from actual images (works for both custom and torchvision datasets)
            input_size, in_channels = self._detect_image_properties(train_folder)
            # Get normalization values based on detected channels
            if in_channels == 1:  # Grayscale
                mean = [0.5]
                std = [0.5]
            elif in_channels == 3:  # Color
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            else:
                logger.warning(f"Unusual channel count {in_channels}, using default normalization")
                mean = [0.5] * in_channels
                std = [0.5] * in_channels

            # Find first valid image
            first_image_path = next(
                (os.path.join(root, f) for root, _, files in os.walk(train_folder)
                 for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))),
                None
            )
            if not first_image_path:
                raise ValueError(f"No valid images found in {train_folder}")

            # Extract image properties
            with Image.open(first_image_path) as img:
                input_size = img.size
                img_tensor = transforms.ToTensor()(img)
                in_channels = img_tensor.shape[0]

            # Count classes
            class_dirs = [d for d in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, d))]
            num_classes = len(class_dirs)
            if num_classes == 0:
                raise ValueError(f"No class directories found in {train_folder}")



            # 1. Generate main JSON config
            json_config = {
                "dataset": {
                    "_comment": "Dataset configuration settings",
                    "name": dataset_name,
                    "_name_comment": "Dataset name",
                    "type": "torchvision" if self.datatype.lower() == "torchvision" else "custom",
                    "_type_comment": "custom or torchvision",
                    "in_channels": in_channels,
                    "_channels_comment": "Number of input channels",
                    "num_classes": num_classes,
                    "_classes_comment": "Number of classes",
                    "input_size": list(input_size),
                    "_size_comment": "Input image dimensions",
                    "mean": mean,
                    "_mean_comment": "Normalization mean",
                    "std": std,
                    "_std_comment": "Normalization standard deviation",
                    "train_dir": train_folder,
                    "_train_comment": "Training data directory",
                    "test_dir": os.path.join(folder_path, 'test'),
                    "_test_comment": "Test data directory"
                },
                "model": {
                    "_comment": "Model configuration settings",
                    "architecture": "CNN",
                    "_arch_comment": "Model architecture type",
                    "feature_dims": 128,
                    "_dims_comment": "Feature dimensions",
                    "learning_rate": 0.001,
                    "_lr_comment": "Learning rate",
                    "optimizer": {
                        "type": "Adam",
                        "_type_comment": "Optimizer type (Adam, SGD)",
                        "weight_decay": 1e-4,
                        "_decay_comment": "Weight decay for regularization",
                        "momentum": 0.9,
                        "_momentum_comment": "Momentum for SGD",
                        "beta1": 0.9,
                        "_beta1_comment": "Beta1 for Adam",
                        "beta2": 0.999,
                        "_beta2_comment": "Beta2 for Adam",
                        "epsilon": 1e-8,
                        "_epsilon_comment": "Epsilon for Adam"
                    },
                    "scheduler": {
                        "type": "StepLR",
                        "_type_comment": "Learning rate scheduler type",
                        "step_size": 7,
                        "_step_comment": "Step size for scheduler",
                        "gamma": 0.1,
                        "_gamma_comment": "Gamma for scheduler"
                    }
                },
                "training": {
                    "_comment": "Training parameters",
                    "batch_size": 32,
                    "_batch_comment": "Batch size for training",
                    "epochs": 20,
                    "_epoch_comment": "Number of epochs",
                    "num_workers": min(4, os.cpu_count() or 1),
                    "_workers_comment": "Number of worker processes",
                    "cnn_training": {
                        "resume": True,
                        "_resume_comment": "Resume from checkpoint if available",
                        "fresh_start": False,
                        "_fresh_comment": "Start training from scratch",
                        "min_loss_threshold": 0.01,
                        "_threshold_comment": "Minimum loss threshold",
                        "checkpoint_dir": "Model/cnn_checkpoints",
                        "_checkpoint_comment": "Checkpoint directory"
                    }
                },
                "execution_flags": {
                    "_comment": "Execution control settings",
                    "mode": "train_and_predict",
                    "_mode_comment": "Execution mode options: train_and_predict, train_only, predict_only",
                    "use_gpu": torch.cuda.is_available(),
                    "_gpu_comment": "Whether to use GPU acceleration",
                    "mixed_precision": True,
                    "_precision_comment": "Whether to use mixed precision training",
                    "distributed_training": False,
                    "_distributed_comment": "Whether to use distributed training",
                    "debug_mode": False,
                    "_debug_comment": "Whether to enable debug mode",
                    "use_previous_model": True,
                    "_prev_model_comment": "Whether to use previously trained model",
                    "fresh_start": False,
                    "_fresh_comment": "Whether to start from scratch"
                }
            }

            # 2. Generate adaptive_dbnn.conf
            dbnn_config = {
                "training_params": {
                    "_comment": "Basic training parameters",
                    "trials": 100,
                    "_trials_comment": "Number of trials",
                    "cardinality_threshold": 0.9,
                    "_card_thresh_comment": "Cardinality threshold",
                    "cardinality_tolerance": 4,
                    "_card_tol_comment": "Cardinality tolerance",
                    "learning_rate": 0.1,
                    "_lr_comment": "Learning rate",
                    "random_seed": 42,
                    "_seed_comment": "Random seed",
                    "epochs": 1000,
                    "_epochs_comment": "Maximum epochs",
                    "test_fraction": 0.2,
                    "_test_frac_comment": "Test data fraction",
                    "enable_adaptive": True,
                    "_adaptive_comment": "Enable adaptive learning",
                    "modelType": "Histogram",
                    "_model_comment": "Model type",
                    "compute_device": "auto",
                    "_device_comment": "Compute device",
                    "use_interactive_kbd": False,
                    "_kbd_comment": "Interactive keyboard",
                    "debug_enabled": True,
                    "_debug_comment": "Debug mode",
                    "Save_training_epochs": True,
                    "_save_comment": "Save training epochs",
                    "training_save_path": f"training_data/{dataset_name}",
                    "_save_path_comment": "Save path"
                },
                "execution_flags": {
                    "train": True,
                    "_train_comment": "Enable training",
                    "train_only": False,
                    "_train_only_comment": "Train only mode",
                    "predict": True,
                    "_predict_comment": "Enable prediction",
                    "gen_samples": False,
                    "_samples_comment": "Generate samples",
                    "fresh_start": False,
                    "_fresh_comment": "Fresh start",
                    "use_previous_model": True,
                    "_prev_model_comment": "Use previous model"
                }
            }

            # 3. Generate dataset-specific .conf
            data_config = {
                "file_path": f"{dataset_name}.csv",
                "_path_comment": "Dataset file path",
                "column_names": [f"feature_{i}" for i in range(128)] + ["target"],
                "_columns_comment": "Column names",
                "separator": ",",
                "_separator_comment": "CSV separator",
                "has_header": True,
                "_header_comment": "Has header row",
                "target_column": "target",
                "_target_comment": "Target column",
                "likelihood_config": {
                    "feature_group_size": 2,
                    "_group_comment": "Feature group size",
                    "max_combinations": 1000,
                    "_combinations_comment": "Max combinations",
                    "bin_sizes": [20],
                    "_bins_comment": "Histogram bin sizes"
                },
                "active_learning": {
                    "tolerance": 1.0,
                    "_tolerance_comment": "Learning tolerance",
                    "cardinality_threshold_percentile": 95,
                    "_percentile_comment": "Cardinality threshold",
                    "strong_margin_threshold": 0.3,
                    "_strong_comment": "Strong margin threshold",
                    "marginal_margin_threshold": 0.1,
                    "_marginal_comment": "Marginal margin threshold",
                    "min_divergence": 0.1,
                    "_divergence_comment": "Minimum divergence"
                },
                "training_params": {
                    "Save_training_epochs": True,
                    "_save_comment": "Save epoch data",
                    "training_save_path": f"training_data/{dataset_name}",
                    "_save_path_comment": "Save path"
                },
                "modelType": "Histogram",
                "_model_comment": "Model type"
            }

            # Return all configurations
            return {
                "json_config": json_config,
                "dbnn_config": dbnn_config,
                "data_config": data_config
            }

        except Exception as e:
            logger.error(f"Error generating configuration: {str(e)}")
            raise

    def _process_custom(self):
        """Process custom dataset from directory or compressed file with improved path handling"""
        if os.path.isdir(self.datafile):
            # Extract dataset name from path
            dataset_name = os.path.basename(os.path.normpath(self.datafile))

            # Create dataset directory in data folder
            dataset_dir = os.path.join('data', dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            # Define train/test directories within dataset directory
            train_dir = os.path.join(dataset_dir, 'train')
            test_dir = os.path.join(dataset_dir, 'test')

            # Check for class subdirectories in input directory
            subdirs = [d for d in os.listdir(self.datafile)
                      if os.path.isdir(os.path.join(self.datafile, d))]

            valid_class_dirs = []
            for subdir in subdirs:
                full_path = os.path.join(self.datafile, subdir)
                if any(f.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS)
                      for f in os.listdir(full_path)):
                    valid_class_dirs.append(subdir)

            if valid_class_dirs:
                logger.info(f"Found {len(valid_class_dirs)} class directories: {valid_class_dirs}")
                response = input("Create train/test split from class directories? (y/n): ").lower()

                if response == 'y':
                    # Create train/test split
                    return self._create_train_test_split_in_data(self.datafile, dataset_dir)
                else:
                    # Use all data for training
                    os.makedirs(train_dir, exist_ok=True)

                    # Copy all class directories to train directory
                    for class_dir in valid_class_dirs:
                        src = os.path.join(self.datafile, class_dir)
                        dst = os.path.join(train_dir, class_dir)
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)

                    logger.info(f"Created training directory structure in {train_dir}")
                    return train_dir, None
            else:
                raise ValueError(f"No valid class directories found in {self.datafile}")

        else:
            raise ValueError(f"Input {self.datafile} is not a valid directory")

    def _create_train_test_split_in_data(self, source_dir: str, dataset_dir: str, test_size: float = 0.2) -> Tuple[str, str]:
        logger.info(f"Creating train/test split with test_size={test_size}")

        # Create train and test directories
        train_dir = os.path.join(dataset_dir, "train")
        test_dir = os.path.join(dataset_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Process each class directory with progress
        class_dirs = [d for d in os.listdir(source_dir)
                     if os.path.isdir(os.path.join(source_dir, d))]

        with tqdm(total=len(class_dirs), desc="Processing classes") as class_pbar:
            for class_name in class_dirs:
                class_path = os.path.join(source_dir, class_name)

                # [Existing code to get image_files...]

                # Create progress bar for file copying
                with tqdm(total=len(train_files)+len(test_files),
                         desc=f"Copying {class_name}", leave=False) as file_pbar:
                    # Copy training files
                    for src in train_files:
                        shutil.copy2(src, os.path.join(train_class_dir, os.path.basename(src)))
                        file_pbar.update(1)

                    # Copy test files
                    for src in test_files:
                        shutil.copy2(src, os.path.join(test_class_dir, os.path.basename(src)))
                        file_pbar.update(1)

                class_pbar.update(1)

        return train_dir, test_dir


    def _find_image_directory(self, start_path: str) -> Tuple[str, str, bool]:
        """
        Find directory containing class subdirectories with images.
        Returns tuple: (dataset_name, directory_path, has_train_test)
        """
        def has_image_subdirs(dir_path):
            """Check if directory has subdirectories containing images"""
            if not os.path.isdir(dir_path):
                return False

            subdirs = [d for d in os.listdir(dir_path)
                      if os.path.isdir(os.path.join(dir_path, d))]

            for subdir in subdirs:
                subdir_path = os.path.join(dir_path, subdir)
                if any(f.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS)
                      for f in os.listdir(subdir_path)):
                    return True
            return False

        # Normalize path
        start_path = os.path.normpath(start_path)
        path_parts = start_path.split(os.sep)

        # First check if this is already a train/test structure
        train_dir = os.path.join(start_path, "train")
        test_dir = os.path.join(start_path, "test")

        if os.path.exists(train_dir) and os.path.exists(test_dir):
            if has_image_subdirs(train_dir) and has_image_subdirs(test_dir):
                # Get dataset name from parent directory that's not 'train' or 'test'
                for i in range(len(path_parts) - 1, -1, -1):
                    if path_parts[i].lower() not in ['train', 'test']:
                        return (path_parts[i], start_path, True)

        # Check if current directory has class subdirectories
        if has_image_subdirs(start_path):
            # Get dataset name from parent directory
            for i in range(len(path_parts) - 1, -1, -1):
                if path_parts[i].lower() not in ['train', 'test']:
                    return (path_parts[i], start_path, False)

        # Recursively search subdirectories
        for root, dirs, _ in os.walk(start_path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)

                # Check for train/test structure
                train_subdir = os.path.join(dir_path, "train")
                test_subdir = os.path.join(dir_path, "test")

                if os.path.exists(train_subdir) and os.path.exists(test_subdir):
                    if has_image_subdirs(train_subdir) and has_image_subdirs(test_subdir):
                        # Get dataset name from parent directory that's not 'train' or 'test'
                        path_parts = os.path.normpath(dir_path).split(os.sep)
                        for i in range(len(path_parts) - 1, -1, -1):
                            if path_parts[i].lower() not in ['train', 'test']:
                                return (path_parts[i], dir_path, True)

                # Check for class subdirectories
                if has_image_subdirs(dir_path):
                    # Get dataset name from parent directory
                    path_parts = os.path.normpath(dir_path).split(os.sep)
                    for i in range(len(path_parts) - 1, -1, -1):
                        if path_parts[i].lower() not in ['train', 'test']:
                            return (path_parts[i], dir_path, False)

        return (None, None, False)
    def process(self):
           """Process dataset with proper directory structure

           Returns:
               Tuple[str, str]: Paths to train and test directories
           """
           # Handle torchvision datasets
           if self.datatype.lower() == 'torchvision':
               logger.info(f"Processing torchvision dataset: {self.datafile}")
               return self._process_torchvision()

           # For custom datasets, find appropriate image directory
           dataset_name, image_dir, has_train_test = self._find_image_directory(self.datafile)

           if dataset_name is None:
               raise ValueError(f"No valid image dataset structure found in {self.datafile}")

           # Update dataset name to be the name of the output directory
           self.dataset_name = os.path.basename(os.path.normpath(self.output_dir))

           # Define train/test directories within dataset directory
           train_dir = os.path.join(self.dataset_dir, 'train')
           test_dir = os.path.join(self.dataset_dir, 'test')

           if has_train_test:
               # Copy existing train/test structure
               src_train = os.path.join(image_dir, 'train')
               src_test = os.path.join(image_dir, 'test')

               if os.path.exists(train_dir):
                   shutil.rmtree(train_dir)
               if os.path.exists(test_dir):
                   shutil.rmtree(test_dir)

               shutil.copytree(src_train, train_dir)
               shutil.copytree(src_test, test_dir)

               logger.info(f"Copied existing train/test structure to {self.dataset_dir}")

           else:
               # Directory has class subdirectories
               class_dirs = [d for d in os.listdir(image_dir)
                            if os.path.isdir(os.path.join(image_dir, d))]

               if not class_dirs:
                   raise ValueError(f"No class directories found in {image_dir}")

               logger.info(f"Found {len(class_dirs)} class directories: {class_dirs}")
               response = input("Create train/test split from class directories? (y/n): ").lower()

               if response == 'y':
                   return self._create_train_test_split_in_data(image_dir, self.dataset_dir)
               else:
                   # Use all data for training
                   os.makedirs(train_dir, exist_ok=True)

                   # Copy all class directories to train directory
                   for class_dir in class_dirs:
                       src = os.path.join(image_dir, class_dir)
                       dst = os.path.join(train_dir, class_dir)
                       if os.path.exists(dst):
                           shutil.rmtree(dst)
                       shutil.copytree(src, dst)

                   logger.info(f"Created training directory structure in {train_dir}")
                   return train_dir, None

           return train_dir, test_dir

    def _load_existing_data(self):
        """Load existing dataset files"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return config['dataset']['train_dir'], config['dataset']['test_dir']


    def _process_torchvision(self):
        """Process torchvision dataset with proper error handling and progress tracking

        Returns:
            Tuple[str, str]: Paths to train and test directories
        """
        try:
            # Convert dataset name to uppercase and verify it exists
            dataset_name = self.datafile.upper()
            if not hasattr(datasets, dataset_name):
                raise ValueError(f"Torchvision dataset {dataset_name} not found.")

            logger.info(f"Downloading and processing {dataset_name} dataset...")

            # Define transforms for image conversion
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

            # Setup output directories
            train_dir = os.path.join(self.output_dir, dataset_name, "train")
            test_dir = os.path.join(self.output_dir, dataset_name, "test")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Download and load datasets with progress tracking
            try:
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
            except Exception as e:
                raise RuntimeError(f"Failed to download {dataset_name} dataset: {str(e)}")

            # Function to save images with proper class directories
            def save_dataset_images(dataset, output_dir, split_name):
                logger.info(f"Processing {split_name} split...")

                # Create mapping of class indices to labels if available
                class_to_idx = getattr(dataset, 'class_to_idx', None)
                if class_to_idx:
                    idx_to_class = {v: k for k, v in class_to_idx.items()}

                with tqdm(total=len(dataset), desc=f"Saving {split_name} images") as pbar:
                    for idx, (img, label) in enumerate(dataset):
                        # Determine class directory name
                        if class_to_idx:
                            class_name = idx_to_class[label]
                        else:
                            class_name = str(label)

                        # Create class directory
                        class_dir = os.path.join(output_dir, class_name)
                        os.makedirs(class_dir, exist_ok=True)

                        # Save image
                        if isinstance(img, torch.Tensor):
                            img = transforms.ToPILImage()(img)

                        img_path = os.path.join(class_dir, f"{idx}.png")
                        img.save(img_path)
                        pbar.update(1)

            # Process and save both splits
            save_dataset_images(train_dataset, train_dir, "training")
            save_dataset_images(test_dataset, test_dir, "test")

            logger.info(f"Successfully processed {dataset_name} dataset")
            logger.info(f"Training images saved to: {train_dir}")
            logger.info(f"Test images saved to: {test_dir}")

            return train_dir, test_dir

        except Exception as e:
            logger.error(f"Error processing torchvision dataset: {str(e)}")
            raise

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
        # Process dataset first
        train_dir, test_dir = self.process()

        # Get dataset name from the processed directory
        dataset_name = os.path.basename(os.path.dirname(train_dir))

        # Generate default configuration using processed directories
        config = self.generate_default_config(os.path.dirname(train_dir))

        # Update config with correct paths
        config['dataset'].update({
            'name': dataset_name,
            'train_dir': train_dir,
            'test_dir': test_dir if test_dir else ''
        })

        return train_dir, test_dir, config

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

    def verify_config(self, config: Dict) -> Dict:
        """
        Verify and fill in missing configuration values with defaults

        Args:
            config: Input configuration dictionary

        Returns:
            Dict: Verified configuration with all required fields
        """
        # Ensure all required sections exist
        required_sections = ['dataset', 'model', 'training', 'execution_flags']
        for section in required_sections:
            if section not in config:
                config[section] = {}

        # Ensure execution_flags has all required fields
        if 'execution_flags' not in config:
            config['execution_flags'] = {}

        exec_flags = config['execution_flags']
        exec_flags.setdefault('mode', 'train_and_predict')
        exec_flags.setdefault('use_gpu', torch.cuda.is_available())
        exec_flags.setdefault('mixed_precision', True)
        exec_flags.setdefault('distributed_training', False)
        exec_flags.setdefault('debug_mode', False)
        exec_flags.setdefault('use_previous_model', True)
        exec_flags.setdefault('fresh_start', False)

        # Ensure training section has required fields
        if 'training' not in config:
            config['training'] = {}

        training = config['training']
        training.setdefault('batch_size', 32)
        training.setdefault('epochs', 20)
        training.setdefault('num_workers', min(4, os.cpu_count() or 1))

        if 'cnn_training' not in training:
            training['cnn_training'] = {}

        cnn_training = training['cnn_training']
        cnn_training.setdefault('resume', True)
        cnn_training.setdefault('fresh_start', False)
        cnn_training.setdefault('min_loss_threshold', 0.01)
        cnn_training.setdefault('checkpoint_dir', 'Model/cnn_checkpoints')

        return config

    def __init__(self, config: Dict, device: str = None):
        """
        Initialize CNN trainer with configuration

        Args:
            config: Configuration dictionary
            device: Optional device specification
        """
        # Verify and complete config
        self.config = self.verify_config(config)

        # Set device
        if device is None:
            self.device = torch.device('cuda' if self.config['execution_flags']['use_gpu']
                                     and torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize other parameters
        self.feature_dims = self.config['model'].get('feature_dims', 128)
        self.learning_rate = self.config['model'].get('learning_rate', 0.001)

        # Initialize model
        self.feature_extractor = FeatureExtractorCNN(
            in_channels=self.config['dataset']['in_channels'],
            feature_dims=self.feature_dims
        ).to(self.device)

        # Configure optimizer
        optimizer_config = self.config['model'].get('optimizer', {})
        optimizer_params = {
            'lr': self.learning_rate,
            'weight_decay': optimizer_config.get('weight_decay', 0)
        }
        if optimizer_config.get('type') == 'SGD' and 'momentum' in optimizer_config:
            optimizer_params['momentum'] = optimizer_config['momentum']

        optimizer_class = getattr(optim, optimizer_config.get('type', 'Adam'))
        self.optimizer = optimizer_class(self.feature_extractor.parameters(), **optimizer_params)

        # Training metrics
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.history = defaultdict(list)
        self.training_log = []

        # Setup logging directory
        self.log_dir = os.path.join('Traininglog', self.config['dataset']['name'])
        os.makedirs(self.log_dir, exist_ok=True)

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
    Get dataset based on configuration. For custom datasets, uses data/<dataset_name>/train structure.
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
        # Use dataset name for folder path
        dataset_name = dataset_config['name']
        base_folder = os.path.join("data", dataset_name)
        train_folder = os.path.join(base_folder, 'train')
        test_folder = os.path.join(base_folder, 'test')

        # Ensure train folder exists
        if not os.path.exists(train_folder):
            raise ValueError(f"Training directory not found at {train_folder}")

        train_dataset = CustomImageDataset(
            data_dir=train_folder,
            transform=transform
        )

        # Only create test dataset if test folder exists
        test_dataset = None
        if os.path.exists(test_folder):
            test_dataset = CustomImageDataset(
                data_dir=test_folder,
                transform=transform
            )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config['type']}")

    if merge_datasets and test_dataset is not None:
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
    def __init__(self, config: Dict, device: str = None):
        """Initialize CNN trainer with configuration"""
        # Verify and complete config
        self.config = self.verify_config(config)

        # Set device
        if device is None:
            self.device = torch.device('cuda' if self.config['execution_flags']['use_gpu']
                                     and torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize model parameters
        self.feature_dims = self.config['model']['feature_dims']
        self.learning_rate = self.config['model']['learning_rate']

        # Initialize training metrics
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.history = defaultdict(list)
        self.training_log = []

        # Setup logging directory
        self.log_dir = os.path.join('Traininglog', self.config['dataset']['name'])
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize model and optimizer
        self.feature_extractor = self._create_model()
        if not self.config['execution_flags'].get('fresh_start', False):
            self._load_from_checkpoint()

        # Initialize optimizer if not already created during checkpoint loading
        if not hasattr(self, 'optimizer'):
            self.optimizer = self._initialize_optimizer()

    def verify_config(self, config: Dict) -> Dict:
        """Verify and fill in missing configuration values with defaults"""
        if 'dataset' not in config:
            raise ValueError("Configuration must contain 'dataset' section")

        # Ensure all required sections exist
        required_sections = ['dataset', 'model', 'training', 'execution_flags']
        for section in required_sections:
            if section not in config:
                config[section] = {}

        # Verify model section
        model = config['model']
        model.setdefault('feature_dims', 128)
        model.setdefault('learning_rate', 0.001)
        model.setdefault('optimizer', {
            'type': 'Adam',
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        })

        # Verify training section
        training = config['training']
        training.setdefault('batch_size', 32)
        training.setdefault('epochs', 20)
        training.setdefault('num_workers', min(4, os.cpu_count() or 1))
        training.setdefault('cnn_training', {
            'resume': True,
            'fresh_start': False,
            'min_loss_threshold': 0.01,
            'checkpoint_dir': 'Model/cnn_checkpoints'
        })

        # Verify execution flags
        exec_flags = config['execution_flags']
        exec_flags.setdefault('mode', 'train_and_predict')
        exec_flags.setdefault('use_gpu', torch.cuda.is_available())
        exec_flags.setdefault('mixed_precision', True)
        exec_flags.setdefault('distributed_training', False)
        exec_flags.setdefault('debug_mode', False)
        exec_flags.setdefault('use_previous_model', True)
        exec_flags.setdefault('fresh_start', False)

        return config

    def _create_model(self) -> nn.Module:
        """Create new model instance"""
        return FeatureExtractorCNN(
            in_channels=self.config['dataset']['in_channels'],
            feature_dims=self.feature_dims
        ).to(self.device)

    def _load_from_checkpoint(self) -> None:
        """Load model and training state from checkpoint if available"""
        checkpoint_path = self._find_latest_checkpoint()
        if checkpoint_path:
            try:
                logger.info(f"Found previous checkpoint at {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                logger.info("Checkpoint loaded successfully")

                # Load model state
                if 'state_dict' in checkpoint:
                    self.feature_extractor.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.feature_extractor.load_state_dict(checkpoint['model_state_dict'])

                # Load training state
                self.current_epoch = checkpoint.get('epoch', 0)
                self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))

                # Initialize and load optimizer
                self.optimizer = self._initialize_optimizer()
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("Optimizer state loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {str(e)}")

                # Load history
                if 'history' in checkpoint:
                    self.history = defaultdict(list, checkpoint['history'])
                    logger.info("Training history loaded")

                logger.info("Successfully initialized model from checkpoint")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                self.optimizer = self._initialize_optimizer()

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer based on configuration"""
        optimizer_config = self.config['model'].get('optimizer', {
            'type': 'Adam',
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        })

        optimizer_params = {
            'lr': self.config['model'].get('learning_rate', 0.001),
            'weight_decay': optimizer_config.get('weight_decay', 1e-4)
        }

        optimizer_type = optimizer_config.get('type', 'Adam')
        if optimizer_type == 'SGD':
            optimizer_params['momentum'] = optimizer_config.get('momentum', 0.9)
        elif optimizer_type == 'Adam':
            optimizer_params['betas'] = (
                optimizer_config.get('beta1', 0.9),
                optimizer_config.get('beta2', 0.999)
            )
            optimizer_params['eps'] = optimizer_config.get('epsilon', 1e-8)

        try:
            optimizer_class = getattr(optim, optimizer_type)
        except AttributeError:
            logger.warning(f"Optimizer {optimizer_type} not found, using Adam")
            optimizer_class = optim.Adam

        return optimizer_class(self.feature_extractor.parameters(), **optimizer_params)

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint for the current dataset"""
        dataset_name = self.config['dataset']['name']
        checkpoint_dir = os.path.join("Model", "cnn_checkpoints")

        if not os.path.exists(checkpoint_dir):
            logger.info(f"No checkpoint directory found at {checkpoint_dir}")
            return None

        # Look for checkpoints
        checkpoints = []

        # Check for best model first
        best_model_path = os.path.join(checkpoint_dir, f"{dataset_name}_best.pth")
        if os.path.exists(best_model_path):
            checkpoints.append(best_model_path)

        # Check for checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoints.append(checkpoint_path)

        if not checkpoints:
            logger.info(f"No checkpoints found for {dataset_name}")
            return None

        # Get latest checkpoint
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        logger.info(f"Latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """Train the model"""
        target_accuracy = self.config['training'].get('target_accuracy', 0.95)
        min_loss = self.config['training'].get('min_loss', 0.01)
        max_epochs = self.config['training']['epochs']
        patience = self.config['training'].get('patience', 5)

        patience_counter = 0
        best_val_metric = float('inf')
        training_start = time.time()

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

            # Log metrics
            self.log_training_metrics(epoch, train_loss, train_acc, val_loss, val_acc,
                                   train_loader, val_loader)

            # Save checkpoint
            self._save_checkpoint(is_best=False)

            # Check for improvement
            if current_metric < best_val_metric:
                best_val_metric = current_metric
                patience_counter = 0
                self._save_checkpoint(is_best=True)
            else:
                patience_counter += 1

            # Check stopping conditions
            if train_acc >= target_accuracy and train_loss <= min_loss:
                logger.info(f"Reached targets: Accuracy {train_acc:.2f}% >= {target_accuracy}%, "
                          f"Loss {train_loss:.4f} <= {min_loss}")
                break

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch with proper resource management"""
        import gc

        self.feature_extractor.train()
        running_loss = 0.0
        correct = 0
        total = 0

        try:
            pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}',
                       unit='batch', leave=False)

            for batch_idx, (inputs, targets) in enumerate(pbar):
                try:
                    # Training step
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.feature_extractor(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    # Update metrics
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

                    # Cleanup current batch
                    del inputs
                    del outputs
                    del loss

                    # Periodic cleanup
                    if batch_idx % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    raise

            pbar.close()
            return running_loss / len(train_loader), 100. * correct / total

        finally:
            # Final cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.feature_extractor.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(val_loader, desc='Validation',
                   unit='batch', leave=False)

        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.feature_extractor(inputs)
                loss = F.cross_entropy(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                val_loss = running_loss / (pbar.n + 1)
                val_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{val_loss:.4f}',
                    'acc': f'{val_acc:.2f}%'
                })

        pbar.close()
        return running_loss / len(val_loader), 100. * correct / total

    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features with proper resource management"""
        import gc

        self.feature_extractor.eval()
        features = []
        labels = []

        try:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="Extracting features")):
                    try:
                        # Move to device and extract features
                        inputs = inputs.to(self.device)
                        outputs = self.feature_extractor(inputs)

                        # Move to CPU and store
                        features.append(outputs.cpu())
                        labels.append(targets)

                        # Cleanup current batch
                        del inputs
                        del outputs

                        # Periodic cleanup
                        if batch_idx % 50 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    except Exception as e:
                        logger.error(f"Error in batch {batch_idx}: {str(e)}")
                        raise

                # Concatenate results
                return torch.cat(features), torch.cat(labels)

        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            raise

        finally:
            # Final cleanup
            del features
            del labels
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_features(self, features: torch.Tensor, labels: torch.Tensor, output_path: str):
        """Save features with proper resource management"""
        import gc

        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Process in chunks to manage memory
            chunk_size = 1000
            total_samples = features.shape[0]

            for start_idx in range(0, total_samples, chunk_size):
                # Get chunk indices
                end_idx = min(start_idx + chunk_size, total_samples)

                # Create chunk dictionary
                feature_dict = {
                    f'feature_{i}': features[start_idx:end_idx, i].numpy()
                    for i in range(features.shape[1])
                }
                feature_dict['target'] = labels[start_idx:end_idx].numpy()

                # Save chunk
                df = pd.DataFrame(feature_dict)
                mode = 'w' if start_idx == 0 else 'a'
                header = start_idx == 0

                with open(output_path, mode, newline='') as f:
                    df.to_csv(f, index=False, header=header)

                # Cleanup chunk data
                del feature_dict
                del df
                gc.collect()

            logger.info(f"Features saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise

        finally:
            # Final cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join('Model', 'cnn_checkpoints')
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

        filename = f"{self.config['dataset']['name']}_{'best' if is_best else 'checkpoint'}.pth"
        path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Saved {'best' if is_best else 'latest'} checkpoint to {path}")

    def log_training_metrics(self, epoch: int, train_loss: float, train_acc: float,
                           test_loss: Optional[float] = None, test_acc: Optional[float] = None,
                           train_loader: Optional[DataLoader] = None, test_loader: Optional[DataLoader] = None):
        """Log training metrics and save to file"""
        # Calculate elapsed time
        if not hasattr(self, 'training_start_time'):
            self.training_start_time = time.time()
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
                      f"Train [{metrics['train_samples']} samples] "
                      f"Loss {train_loss:.4f}, Acc {train_acc:.2f}%")

        if test_loss is not None:
            log_message += (f", Test [{metrics['test_samples']} samples] "
                          f"Loss {test_loss:.4f}, Acc {test_acc:.2f}%")

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

    def plot_confusion_matrix(self, loader: DataLoader, save_path: Optional[str] = None):
        """Plot confusion matrix for the given data loader"""
        self.feature_extractor.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Computing predictions"):
                inputs = inputs.to(self.device)
                outputs = self.feature_extractor(inputs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())

        # Compute confusion matrix
        cm = confusion_matrix(all_targets, all_preds)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        plt.close()

    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, lr: float):
        """Set learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        logger.info(f"Learning rate set to {lr}")

    def get_device(self) -> torch.device:
        """Get current device"""
        return self.device

    def to(self, device: torch.device):
        """Move model to device"""
        self.device = device
        self.feature_extractor.to(device)
        logger.info(f"Model moved to {device}")

    def get_model_summary(self) -> str:
        """Get model summary"""
        total_params = sum(p.numel() for p in self.feature_extractor.parameters())
        trainable_params = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)

        return (f"Model Summary:\n"
                f"Architecture: {self.feature_extractor.__class__.__name__}\n"
                f"Total parameters: {total_params:,}\n"
                f"Trainable parameters: {trainable_params:,}\n"
                f"Device: {self.device}\n"
                f"Current learning rate: {self.get_learning_rate()}\n"
                f"Current epoch: {self.current_epoch}\n"
                f"Best accuracy: {self.best_accuracy:.2f}%\n"
                f"Best loss: {self.best_loss:.4f}")

    @staticmethod
    def load_model_from_checkpoint(checkpoint_path: str, config: Dict = None) -> 'CNNTrainer':
        """Load model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get config from checkpoint or use provided config
        if config is None:
            if 'config' not in checkpoint:
                raise ValueError("No configuration found in checkpoint and none provided")
            config = checkpoint['config']

        # Create trainer instance
        trainer = CNNTrainer(config)

        # Load state
        trainer._load_from_checkpoint()

        return trainer

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
    print("  --config        Path to configuration file (overrides other options)")
    print("  --batch_size    Batch size for training (default: 32)")
    print("  --epochs        Number of training epochs (default: 20)")
    print("  --workers       Number of data loading workers (default: 4)")
    print("  --learning_rate Learning rate (default: 0.001)")
    print("  --output-dir    Output directory (default: data)")
    print("  --cpu          Force CPU usage even if GPU is available")
    print("  --debug        Enable debug mode with verbose logging")
    print("  --merge-datasets Merge train and test datasets")

    print("\nExamples:")
    print("  1. Process MNIST dataset from torchvision:")
    print("     python cdbnn.py --data_type torchvision --data MNIST")

    print("\n  2. Process custom image dataset:")
    print("     python cdbnn.py --data_type custom --data path/to/images")

    print("\n  3. Use configuration file:")
    print("     python cdbnn.py --config config.json")

    print("\n  4. Customize training parameters:")
    print("     python cdbnn.py --data_type custom --data path/to/images \\")
    print("                     --batch_size 64 --epochs 50 --learning_rate 0.0001")

    print("\nDirectory Structure for Custom Datasets:")
    print("  Option 1 - Pre-split data:")
    print("    dataset_folder/")
    print("    ├── train/")
    print("    │   ├── class1/")
    print("    │   ├── class2/")
    print("    │   └── ...")
    print("    └── test/")
    print("        ├── class1/")
    print("        ├── class2/")
    print("        └── ...")

    print("\n  Option 2 - Single directory:")
    print("    dataset_folder/")
    print("    ├── class1/")
    print("    ├── class2/")
    print("    └── ...")
    print("    (Will be prompted to create train/test split)")
def parse_arguments():
    """Parse command line arguments, return None if no arguments provided"""
    if len(sys.argv) == 1:
        return None

    parser = argparse.ArgumentParser(description='CNN Feature Extractor Training')

    # Dataset options
    parser.add_argument('--data_type', type=str, choices=['torchvision', 'custom'],
                      help='type of dataset (torchvision or custom)')
    parser.add_argument('--data', type=str,
                      help='dataset name for torchvision or path for custom dataset')

    # Configuration
    parser.add_argument('--config', type=str,
                      help='path to configuration file (overrides other options)')

    # Training parameters
    parser.add_argument('--batch_size', type=int,
                      help='batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int,
                      help='number of training epochs (default: 20)')
    parser.add_argument('--workers', type=int,
                      help='number of data loading workers (default: 4)')
    parser.add_argument('--learning_rate', type=float,
                      help='learning rate (default: 0.001)')

    # Output and execution options
    parser.add_argument('--output-dir', type=str, default='data',
                      help='output directory (default: data)')
    parser.add_argument('--cpu', action='store_true',
                      help='force CPU usage')
    parser.add_argument('--debug', action='store_true',
                      help='enable debug mode')

    return parser.parse_args()

def main():
    """Main execution function supporting both interactive and command line modes"""
    try:
        # Set up logging
        logger = setup_logging()
        logger.info("Starting CNN training process...")

        # Get arguments
        args = parse_arguments()
        config = None

        # Handle interactive mode if no arguments provided
        if args is None:
            print("\nEntering interactive mode...")
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

            # Get optional parameters
            args.batch_size = int(input("Enter batch size (default: 32): ").strip() or "32")
            args.epochs = int(input("Enter number of epochs (default: 20): ").strip() or "20")
            args.output_dir = input("Enter output directory (default: data): ").strip() or "data"

            # Set defaults for other arguments
            args.workers = None
            args.learning_rate = None
            args.cpu = False
            args.debug = False
            args.config = None

        # Load configuration if provided
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading configuration file: {str(e)}")
                return 1

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
            config_dict = processor.generate_default_config(os.path.dirname(train_dir))
            config = config_dict["json_config"]

        # Update config with command line arguments if provided
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
        if args.debug:
            config['execution_flags']['debug_mode'] = True

        # Initialize trainer
        logger.info("Initializing CNN trainer...")
        device = torch.device('cuda' if config['execution_flags']['use_gpu']
                            and torch.cuda.is_available() else 'cpu')
        trainer = CNNTrainer(config=config, device=device)

        # Get transforms
        transform = processor.get_transforms(config)

        # Prepare datasets
        train_dataset, test_dataset = get_dataset(config, transform)
        if train_dataset is None:
            raise ValueError("No training dataset available")

        # Create data loaders
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

        # Train model
        logger.info("Starting model training...")
        history = trainer.train(train_loader, test_loader)

        # Extract features
        logger.info("Extracting features...")
        train_features, train_labels = trainer.extract_features(train_loader)

        if test_loader:
            test_features, test_labels = trainer.extract_features(test_loader)
            features = torch.cat([train_features, test_features])
            labels = torch.cat([train_labels, test_labels])
        else:
            features = train_features
            labels = train_labels

        # Save features
        output_path = os.path.join(args.output_dir, config['dataset']['name'],
                                 f"{config['dataset']['name']}.csv")
        trainer.save_features(features, labels, output_path)
        logger.info(f"Features saved to {output_path}")

        # Plot training history
        if history:
            plot_path = os.path.join(args.output_dir, config['dataset']['name'],
                                   'training_history.png')
            plot_training_history(history, plot_path)
            logger.info(f"Training history plot saved to {plot_path}")

        logger.info("Processing completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        if hasattr(args, 'debug') and args.debug:
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Feature Extractor Training',
                                   usage='%(prog)s [options]')

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
    parser.add_argument('--output-dir', type=str, default='data',
                        help='output directory (default: data)')
    parser.add_argument('--cpu', action='store_true',
                        help='force CPU usage')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode')
    parser.add_argument('--merge-datasets', action='store_true',
                        help='merge train and test datasets')

    args = parser.parse_args()
    sys.exit(main(args))
