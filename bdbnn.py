import os
import sys
import gc
import json
import logging
import torch
import random
import pandas as pd
import numpy as np
import torchvision
import zipfile
import tarfile
import shutil
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path
from datetime import datetime
import pickle
import argparse
import traceback
import subprocess
import psutil
from itertools import combinations
import torch.multiprocessing
# Set sharing strategy at the start
torch.multiprocessing.set_sharing_strategy('file_system')
# Import the CDBNN and ADBNN modules
from cdbnn import CNNTrainer, DatasetProcessor, ConfigManager
from adbnn import DBNN, DatasetConfig,BinWeightUpdater

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class BDBNNVisualizer:
    """Class for creating visualizations of BDBNN results"""

    def __init__(self, base_dir: Path, config: Dict):
        """
        Initialize visualizer with base directory and configuration.

        Args:
            base_dir: Base directory for visualizations
            config: Configuration dictionary
        """
        self.base_dir = base_dir
        self.config = config
        self.viz_dir = base_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def _get_size_mapping(self, data: pd.DataFrame, target_column: str) -> pd.Series:
        """Create size mapping based on class frequencies."""
        class_counts = data[target_column].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()

        size_mapping = {}
        for class_label, count in class_counts.items():
            if max_count == min_count:
                size = 10
            else:
                size = 8 + (7 * (max_count - count) / (max_count - min_count))
            size_mapping[class_label] = float(size)

        sizes = data[target_column].map(size_mapping)
        return sizes.fillna(8.0)

    def create_epoch_visualizations(self, data: pd.DataFrame, epoch: int,
                                  set_type: str, target_column: str):
        """Create visualizations for a specific epoch and dataset."""
        epoch_viz_dir = self.viz_dir / f'epoch_{epoch}'
        epoch_viz_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data
        data = data.reset_index(drop=True)
        feature_cols = [col for col in data.columns if col != target_column]
        point_sizes = self._get_size_mapping(data, target_column)

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[feature_cols])

        # 1. t-SNE 2D
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=min(30, len(data) - 1))
        tsne_result = tsne.fit_transform(scaled_features)

        tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
        tsne_df[target_column] = data[target_column]

        fig_2d = go.Figure()
        for class_label in sorted(data[target_column].unique()):
            mask = tsne_df[target_column] == class_label
            size_value = float(point_sizes[mask.index[mask]].iloc[0])

            fig_2d.add_trace(go.Scatter(
                x=tsne_df.loc[mask, 'TSNE1'],
                y=tsne_df.loc[mask, 'TSNE2'],
                mode='markers',
                name=f'Class {class_label}',
                marker=dict(
                    size=size_value,
                    line=dict(width=0.5, color='DarkSlateGrey'),
                    opacity=0.7
                )
            ))

        fig_2d.update_layout(title=f't-SNE 2D Projection - {set_type} set')
        fig_2d.write_html(str(epoch_viz_dir / f'tsne_2d_{set_type}.html'))

        # 2. t-SNE 3D
        tsne = TSNE(n_components=3, random_state=42,
                    perplexity=min(30, len(data) - 1))
        tsne_result = tsne.fit_transform(scaled_features)

        tsne_df = pd.DataFrame(tsne_result,
                              columns=['TSNE1', 'TSNE2', 'TSNE3'])
        tsne_df[target_column] = data[target_column]

        fig_3d = go.Figure()
        for class_label in sorted(data[target_column].unique()):
            mask = tsne_df[target_column] == class_label
            size_value = float(point_sizes[mask.index[mask]].iloc[0])

            fig_3d.add_trace(go.Scatter3d(
                x=tsne_df.loc[mask, 'TSNE1'],
                y=tsne_df.loc[mask, 'TSNE2'],
                z=tsne_df.loc[mask, 'TSNE3'],
                mode='markers',
                name=f'Class {class_label}',
                marker=dict(
                    size=size_value,
                    line=dict(width=0.5, color='DarkSlateGrey'),
                    opacity=0.7
                )
            ))

        fig_3d.update_layout(title=f't-SNE 3D Projection - {set_type} set')
        fig_3d.write_html(str(epoch_viz_dir / f'tsne_3d_{set_type}.html'))

        # 3. Feature Combinations (if enough features)
        if len(feature_cols) >= 3:
            feature_combinations = list(combinations(feature_cols, 3))
            max_combinations = 10
            if len(feature_combinations) > max_combinations:
                feature_combinations = feature_combinations[:max_combinations]

            for i, (f1, f2, f3) in enumerate(feature_combinations):
                fig_3d_feat = go.Figure()

                for class_label in sorted(data[target_column].unique()):
                    mask = data[target_column] == class_label
                    size_value = float(point_sizes[mask.index[mask]].iloc[0])

                    fig_3d_feat.add_trace(go.Scatter3d(
                        x=data.loc[mask, f1],
                        y=data.loc[mask, f2],
                        z=data.loc[mask, f3],
                        mode='markers',
                        name=f'Class {class_label}',
                        marker=dict(
                            size=size_value,
                            line=dict(width=0.5, color='DarkSlateGrey'),
                            opacity=0.7
                        )
                    ))

                fig_3d_feat.update_layout(
                    title=f'Features: {f1}, {f2}, {f3} - {set_type} set',
                    scene=dict(
                        xaxis_title=f1,
                        yaxis_title=f2,
                        zaxis_title=f3
                    )
                )
                fig_3d_feat.write_html(
                    str(epoch_viz_dir / f'features_3d_{i+1}_{set_type}.html'))

        # 4. Parallel Coordinates
        fig_parallel = px.parallel_coordinates(
            data, dimensions=feature_cols,
            color=target_column,
            title=f'Parallel Coordinates - {set_type} set'
        )
        fig_parallel.write_html(
            str(epoch_viz_dir / f'parallel_coords_{set_type}.html'))

        # 5. Correlation Matrix
        corr_matrix = data[feature_cols + [target_column]].corr()
        fig_corr = px.imshow(
            corr_matrix,
            title=f'Correlation Matrix - {set_type} set',
            aspect='auto'
        )
        fig_corr.write_html(
            str(epoch_viz_dir / f'correlation_matrix_{set_type}.html'))

def create_default_configs(input_path: str, args=None) -> Tuple[str, str]:
    """Create configuration files based on actual dataset properties"""

    # Initialize dataset handler
    handler = DatasetHandler()

    try:
        # Analyze dataset
        dataset_props = handler.analyze_dataset(input_path)
        dataset_name = dataset_props['name']

        # Define config paths - ensure they're in data/<dataset_name> folder
        config_dir = Path('data') / dataset_name
        config_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        cnn_config_path = config_dir / f"{dataset_name}.conf"
        dbnn_config_path = config_dir / "adaptive_dbnn.conf"

        # Print paths for debugging
        print(f"\nCreating configuration files:")
        print(f"Dataset config path: {cnn_config_path}")
        print(f"DBNN config path: {dbnn_config_path}")

        # Check if configs already exist
        if cnn_config_path.exists() and dbnn_config_path.exists():
            print(f"Found existing configuration files in {config_dir}")
            return str(cnn_config_path), str(dbnn_config_path)

        print("Creating new configuration files...")

        # Create CNN config
        cnn_config = {
            "dataset": {
                "_comment": "Dataset configuration",
                "name": dataset_name,
                "type": dataset_props['type'],
                "in_channels": dataset_props['in_channels'],
                "num_classes": dataset_props['num_classes'],
                "input_size": dataset_props['input_size'],
                "mean": dataset_props['mean'],
                "std": dataset_props['std'],
                "train_dir": dataset_props['train_dir'],
                "test_dir": dataset_props['test_dir'],
                "classes": dataset_props['classes']
            },
            "model": {
                "_comment": "Model architecture settings",
                "feature_dims": args.feature_dims if args and args.feature_dims else 128,
                "learning_rate": args.learning_rate if args and args.learning_rate else 0.001,
                "optimizer": {
                    "type": "Adam",
                    "weight_decay": 1e-4,
                    "momentum": 0.9,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-8
                },
                "scheduler": {
                    "type": "StepLR",
                    "step_size": 7,
                    "gamma": 0.1
                }
            },
            "training": {
                "_comment": "Training parameters",
                "batch_size": args.batch_size if args and args.batch_size else 32,
                "epochs": args.epochs if args and args.epochs else 20,
                "num_workers": args.workers if args and args.workers else min(4, os.cpu_count() or 1),
                "cnn_training": {
                    "resume": True,
                    "fresh_start": args.fresh if args and args.fresh else False,
                    "min_loss_threshold": 0.01,
                    "checkpoint_dir": "Model/cnn_checkpoints"
                },
                "save_training_epochs": True,
                "training_save_path": str(Path("training_data") / dataset_name)
            },
            "augmentation": {
                "_comment": "Data augmentation settings",
                "enable": True,
                "random_resized_crop": {
                    "enable": True,
                    "size": dataset_props['input_size'],
                    "scale": [0.08, 1.0],
                    "ratio": [0.75, 1.33]
                },
                "random_horizontal_flip": {
                    "enable": True,
                    "p": 0.5
                },
                "color_jitter": {
                    "enable": True,
                    "brightness": 0.4,
                    "contrast": 0.4,
                    "saturation": 0.4,
                    "hue": 0.1
                },
                "normalize": {
                    "enable": True,
                    "mean": dataset_props['mean'],
                    "std": dataset_props['std']
                }
            },
            "execution_flags": {
                "_comment": "Execution control flags",
                "mode": "train_and_predict",
                "use_gpu": args.use_gpu if args and hasattr(args, 'use_gpu') else torch.cuda.is_available(),
                "mixed_precision": True,
                "distributed_training": False,
                "debug_mode": args.debug if args and args.debug else False,
                "fresh_start": args.fresh if args and args.fresh else False
            }
        }

        # Create DBNN config
        dbnn_config = {
            "training_params": {
                "_comment": "Basic training parameters",
                "trials": args.trials if args and args.trials else 100,
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "learning_rate": args.dbnn_lr if args and args.dbnn_lr else 0.1,
                "random_seed": 42,
                "epochs": args.dbnn_epochs if args and args.dbnn_epochs else 1000,
                "test_fraction": args.val_split if args and args.val_split else 0.2,
                "enable_adaptive": True,
                "modelType": args.model_type if args and args.model_type else "Histogram",
                "compute_device": "auto",
                "use_interactive_kbd": False,
                "debug_enabled": args.debug if args and args.debug else False,
                "Save_training_epochs": True,
                "training_save_path": str(Path("training_data") / dataset_name)
            },
            "likelihood_config": {
                "_comment": "Likelihood computation settings",
                "feature_group_size": 2,
                "max_combinations": 1000,
                "bin_sizes": [20]
            },
            "active_learning": {
                "_comment": "Active learning parameters",
                "tolerance": 1.0,
                "cardinality_threshold_percentile": 95,
                "strong_margin_threshold": 0.3,
                "marginal_margin_threshold": 0.1,
                "min_divergence": 0.1
            },
            "execution_flags": {
                "_comment": "Execution control flags",
                "train": True,
                "train_only": False,
                "predict": True,
                "gen_samples": False,
                "fresh_start": False,
                "use_previous_model": True
            }
        }

        # Save configurations with proper error handling
        try:
            # Save CNN config
            with open(cnn_config_path, 'w') as f:
                json.dump(cnn_config, f, indent=4)
                print(f"Created CNN configuration file: {cnn_config_path}")

            # Save DBNN config
            with open(dbnn_config_path, 'w') as f:
                json.dump(dbnn_config, f, indent=4)
                print(f"Created DBNN configuration file: {dbnn_config_path}")

        except Exception as e:
            print(f"Error saving configuration files: {str(e)}")
            # Clean up any partially created files
            if os.path.exists(cnn_config_path):
                os.remove(cnn_config_path)
            if os.path.exists(dbnn_config_path):
                os.remove(dbnn_config_path)
            raise

        return str(cnn_config_path), str(dbnn_config_path)

    except Exception as e:
        print(f"Error creating configurations: {str(e)}")
        traceback.print_exc()
        raise

def run_cdbnn_process(dataset_path: str, dataset_type: str, config_path: str = None, use_gpu: bool = True, debug: bool = False) -> bool:
    """Run CDBNN process with proper arguments

    Args:
        dataset_path: Path to dataset
        dataset_type: Type of dataset (torchvision/custom)
        config_path: Optional path to configuration file
        use_gpu: Whether to use GPU
        debug: Whether to enable debug mode

    Returns:
        bool: True if process succeeded, False otherwise
    """
    try:
        # Build command line arguments
        cmd = [
            "python",
            "cdbnn.py",
            "--data_type", dataset_type,
            "--data", dataset_path
        ]

        if config_path:
            cmd.extend(["--config", config_path])
        if not use_gpu:
            cmd.append("--cpu")
        if debug:
            cmd.append("--debug")

        # Run CDBNN with real-time output
        print("\nRunning CDBNN processing with command:", ' '.join(cmd))

        # Use subprocess.Popen to get real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            universal_newlines=True,
            bufsize=1
        )

        # Process and display output in real-time
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
                output_lines.append(line)

        # Wait for process to complete and get return code
        return_code = process.wait()

        if return_code != 0:
            print("\nError in CDBNN processing - non-zero return code:", return_code)
            return False

        if debug:
            print("\nComplete CDBNN Output:")
            print(''.join(output_lines))

        return True

    except Exception as e:
        print(f"\nError running CDBNN: {str(e)}")
        if debug:
            traceback.print_exc()
        return False

class BDBNNConfig:
    """Configuration manager for BDBNN following similar pattern to ADBNN"""

    def __init__(self, config_path: str):
        """Initialize with config file path"""
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load and validate configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Validate required sections
            required_sections = ['dataset', 'model', 'training', 'augmentation', 'execution_flags']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")

            # Set default values where needed
            config['model'].setdefault('feature_dims', 128)
            config['training'].setdefault('batch_size', 32)
            config['training'].setdefault('num_workers', 4)

            return config
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            raise

    @property
    def dataset_config(self) -> dict:
        """Get dataset configuration"""
        return self.config['dataset']

    @property
    def model_config(self) -> dict:
        """Get model configuration"""
        return self.config['model']

    @property
    def training_config(self) -> dict:
        """Get training configuration"""
        return self.config['training']

    @property
    def augmentation_config(self) -> dict:
        """Get augmentation configuration"""
        return self.config['augmentation']

    @property
    def execution_flags(self) -> dict:
        """Get execution flags"""
        return self.config['execution_flags']

    def get_transforms_config(self) -> dict:
        """Get transforms configuration for DatasetProcessor"""
        return {
            'augmentation': self.augmentation_config,
            'input_size': self.dataset_config['input_size'],
            'mean': self.dataset_config['mean'],
            'std': self.dataset_config['std']
        }

class BDBNNBridge:
    """Bridge module with proper config handling"""

    def __init__(self, config_path: str, base_dir: str = "bridge_workspace"):
        """Initialize with config file path"""
        self.config = BDBNNConfig(config_path)
        self.base_dir = Path(base_dir)
        self._setup_directories()
        self.setup_logging()

        # Initialize processor with config
        self.processor = DatasetProcessor(
            dataset_path=self.config.dataset_config['train_dir'],
            dataset_type=self.config.dataset_config['type']
        )

        # Initialize CNN trainer with config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn_trainer = CNNTrainer(config=self.config.model_config, device=device)

    def _setup_directories(self):
        """Setup required directories"""
        # Create main directories
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.config_dir = self.base_dir / "configs"
        self.data_dir = self.base_dir / "data"  # align with cdbnn structure
        self.model_dir = self.base_dir / "Model" # align with cdbnn structure
        self.viz_dir = self.base_dir / "visualizations"

        # Create all directories
        for directory in [self.config_dir, self.data_dir, self.model_dir, self.viz_dir]:
            directory.mkdir(exist_ok=True)

    def setup_logging(self):
        """Configure logging for the bridge module"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f"bdbnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("BDBNN")

    def process_dataset(self,
                       dataset_path: str,
                       dataset_type: str = "custom",
                       config_path: Optional[str] = None) -> Dict:
        """
        Process a dataset through both CDBNN and ADBNN pipelines.

        Args:
            dataset_path: Path to the dataset
            dataset_type: Type of dataset ('custom' or 'torchvision')
            config_path: Optional path to existing configuration

        Returns:
            Dictionary containing processing results
        """
        try:
            self.logger.info(f"Starting dataset processing: {dataset_path}")

            # Step 1: Initialize CDBNN components
            processor = DatasetProcessor(dataset_path, dataset_type)
            config_manager = ConfigManager(str(self.config_dir))

            # Step 2: Process dataset and get configuration
            train_dir, test_dir, cnn_config = processor.process_with_config()

            # Step 3: Generate and save configurations
            self.current_config = self._generate_unified_config(cnn_config)
            config_path = self._save_configurations()

            # Step 4: Initialize CNN trainer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.cnn_trainer = CNNTrainer(config=self.current_config, device=device)

            # Step 5: Extract features
            features_path = self._extract_and_save_features(train_dir, test_dir)

            # Step 6: Initialize and train DBNN
            self.dbnn_classifier = self._initialize_dbnn(features_path)

            # Step 7: Run complete pipeline
            results = self._run_pipeline()

            # Step 8: Initialize visualizer
            self.visualizer = BDBNNVisualizer(self.base_dir, self.current_config)

            # Step 9: Generate visualizations for each epoch
            self._generate_visualizations()

            return results

        except Exception as e:
            self.logger.error(f"Error in dataset processing: {str(e)}", exc_info=True)
            raise

    def _generate_unified_config(self, cnn_config: Dict) -> Dict:
        """Generate unified configuration for both CNN and DBNN"""
        dataset_name = cnn_config['dataset']['name']

        # Construct paths according to cdbnn structure
        dataset_data_dir = self.data_dir / dataset_name
        dataset_data_dir.mkdir(exist_ok=True)

        unified_config = {
            **cnn_config,
            "bridge_config": {
                "intermediate_path": str(dataset_data_dir / f"{dataset_name}_features.csv"),
                "final_output_path": str(dataset_data_dir / f"{dataset_name}_predictions.csv"),
                "checkpoint_dir": str(self.model_dir / dataset_name),
                "config_path": str(self.data_dir / f"{dataset_name}.conf")  # align with cdbnn
            },
            "dbnn_config": {
                "file_path": str(dataset_data_dir / f"{dataset_name}_features.csv"),
                "target_column": "target",
                "modelType": "Histogram",
                "training_params": {
                    "trials": 100,
                    "epochs": 1000,
                    "learning_rate": 0.1,
                    "enable_adaptive": True,
                    "training_save_path": str(dataset_data_dir)  # for adaptive training data
                }
            }
        }

        return unified_config

    def _save_configurations(self) -> str:
        """Save all configurations to disk following cdbnn structure"""
        if not self.current_config:
            raise ValueError("No configuration available to save")

        dataset_name = self.current_config['dataset']['name']

        # Save main config in data directory (cdbnn structure)
        config_path = self.data_dir / f"{dataset_name}.conf"

        # Save cdbnn specific config
        cnn_config = {
            'dataset': self.current_config['dataset'],
            'model': self.current_config['model'],
            'training': self.current_config['training'],
            'augmentation': self.current_config.get('augmentation', {}),
            'execution_flags': self.current_config['execution_flags']
        }

        # Save dbnn specific config
        dbnn_config = self.current_config['dbnn_config']
        dbnn_config_path = self.data_dir / f"{dataset_name}_dbnn.conf"

        # Save the configs
        with open(config_path, 'w') as f:
            json.dump(cnn_config, f, indent=4)

        with open(dbnn_config_path, 'w') as f:
            json.dump(dbnn_config, f, indent=4)

        # Also save unified config for reference
        unified_config_path = self.config_dir / f"{dataset_name}_unified_config.json"
        with open(unified_config_path, 'w') as f:
            json.dump(self.current_config, f, indent=4)

        self.logger.info(f"Saved configurations to {self.data_dir}")
        return str(config_path)

    def _extract_and_save_features(self, train_dir: str, test_dir: str) -> str:
        """Extract features using CNN and save them following cdbnn structure"""
        self.logger.info("Starting feature extraction")

        dataset_name = self.current_config['dataset']['name']
        dataset_data_dir = self.data_dir / dataset_name
        features_path = dataset_data_dir / f"{dataset_name}.csv"  # align with cdbnn

        # Process features using cnn_trainer and save
        processor = DatasetProcessor(train_dir, "custom")
        transform = self.processor.get_transforms(self.config.get_transforms_config())
        train_dataset, test_dataset = self.processor.get_dataset(
            self.config.dataset_config,
            transform
        )

        train_features, train_labels = self.cnn_trainer.extract_features(train_dataset)
        if test_dataset:
            test_features, test_labels = self.cnn_trainer.extract_features(test_dataset)
            all_features = torch.cat([train_features, test_features])
            all_labels = torch.cat([train_labels, test_labels])
        else:
            all_features = train_features
            all_labels = train_labels

        # Save features
        feature_dict = {
            f'feature_{i}': all_features[:, i].numpy()
            for i in range(all_features.shape[1])
        }
        feature_dict['target'] = all_labels.numpy()

        df = pd.DataFrame(feature_dict)
        df.to_csv(features_path, index=False)

        self.logger.info(f"Saved extracted features to {features_path}")
        return str(features_path)

    def _initialize_dbnn(self, features_path: str) -> DBNN:
        """Initialize DBNN with extracted features"""
        self.logger.info("Initializing DBNN classifier")

        # Create DBNN configuration
        dbnn_config = {
            **self.current_config['dbnn_config'],
            'file_path': features_path
        }

        # Initialize DBNN
        dataset_name = self.current_config['dataset']['name']
        dbnn = DBNN(dataset_name)

        return dbnn

    def _run_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline"""
        self.logger.info("Starting complete pipeline execution")

        try:
            # Train DBNN
            results = self.dbnn_classifier.fit_predict()

            # Save final predictions
            predictions_path = self.current_config['bridge_config']['final_output_path']
            self.dbnn_classifier.save_predictions(
                self.dbnn_classifier.data,
                results['predictions'],
                predictions_path
            )

            # Save training indices for visualization
            self._save_training_indices(results.get('train_indices', []),
                                      results.get('test_indices', []))

            pipeline_results = {
                'cnn_features_path': self.current_config['bridge_config']['intermediate_path'],
                'final_predictions_path': predictions_path,
                'accuracy': results.get('accuracy', None),
                'classification_report': results.get('classification_report', None),
                'training_history': results.get('training_history', None)
            }

            return pipeline_results

        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {str(e)}", exc_info=True)
            raise

    def _save_training_indices(self, train_indices: List[int], test_indices: List[int]):
        """Save training and test indices for visualization"""
        epoch_dir = self.model_dir / 'epoch_0'
        epoch_dir.mkdir(exist_ok=True)

        model_type = self.current_config['dbnn_config'].get('modelType', 'Histogram')

        # Save indices
        with open(epoch_dir / f'{model_type}_train_indices.pkl', 'wb') as f:
            pickle.dump(train_indices, f)
        with open(epoch_dir / f'{model_type}_test_indices.pkl', 'wb') as f:
            pickle.dump(test_indices, f)

    def _generate_visualizations(self):
        """Generate visualizations for the processed data"""
        if not self.visualizer:
            self.logger.warning("Visualizer not initialized. Skipping visualization generation.")
            return

        try:
            # Load the processed data
            features_path = self.current_config['bridge_config']['intermediate_path']
            data = pd.read_csv(features_path)

            # Get train/test indices
            epoch_dir = self.model_dir / 'epoch_0'
            model_type = self.current_config['dbnn_config'].get('modelType', 'Histogram')

            with open(epoch_dir / f'{model_type}_train_indices.pkl', 'rb') as f:
                train_indices = pickle.load(f)
            with open(epoch_dir / f'{model_type}_test_indices.pkl', 'rb') as f:
                test_indices = pickle.load(f)

            # Split data into train and test
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]

            # Generate visualizations
            target_column = self.current_config['dbnn_config']['target_column']
            self.visualizer.create_epoch_visualizations(
                train_data, 0, 'train', target_column)
            self.visualizer.create_epoch_visualizations(
                test_data, 0, 'test', target_column)

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}", exc_info=True)

    def process_new_data(self, data_path: str) -> Dict[str, Any]:
        """
        Process new data using the trained models

        Args:
            data_path: Path to new data to be processed

        Returns:
            Dictionary containing processing results
        """
        if not (self.cnn_trainer and self.dbnn_classifier):
            raise RuntimeError("Models not initialized. Run process_dataset first.")

        try:
            # Extract features from new data
            processor = DatasetProcessor(data_path, "custom")
            transform = processor.get_transforms(self.current_config)
            new_dataset = processor.get_dataset(self.current_config, transform)[0]

            features = self.cnn_trainer.extract_features(new_dataset)

            # Save features in temporary file
            temp_features_path = self.data_dir / "temp_features.csv"
            feature_dict = {
                f'feature_{i}': features[:, i].numpy()
                for i in range(features.shape[1])
            }
            pd.DataFrame(feature_dict).to_csv(temp_features_path, index=False)

            # Get predictions from DBNN
            predictions = self.dbnn_classifier.predict(temp_features_path)

            return {
                'predictions': predictions,
                'features_path': str(temp_features_path)
            }

        except Exception as e:
            self.logger.error(f"Error processing new data: {str(e)}", exc_info=True)
            raise

def get_dataset_info() -> Tuple[str, str]:
    """Get dataset path and type from user input"""
    print("\n=== Dataset Information ===")
    dataset_type = input("Please enter the  type of your image dataset (torchvision or custom) :").strip().lower() or "torchvision"
    dataset_path = input("Please enter the path to your image dataset: ").strip() or "mnist"
    if dataset_type =="torchvision":
        dataset_path=dataset_path.upper()

    while not os.path.exists(dataset_path) and dataset_type !="torchvision":
        print("Error: The specified path does not exist.")
        dataset_type = input("Please enter the  type of your image dataset (torchvision or custom) :").strip().lower() or "torchvision"
        dataset_path = input("Please enter a valid path to your image dataset: ").strip()

    return dataset_path, dataset_type

def run_cdbnn_process(dataset_path: str, dataset_type: str, config_path: str = None, use_gpu: bool = True, debug: bool = False) -> bool:
    """Run CDBNN process with proper arguments and handle interactive I/O

    Args:
        dataset_path: Path to dataset
        dataset_type: Type of dataset (torchvision/custom)
        config_path: Optional path to configuration file
        use_gpu: Whether to use GPU
        debug: Whether to enable debug mode

    Returns:
        bool: True if process succeeded, False otherwise
    """
    try:
        # Build command line arguments
        cmd = [
            "python",
            "cdbnn.py",
            "--data_type", dataset_type,
            "--data", dataset_path
        ]

        if config_path:
            cmd.extend(["--config", config_path])
        if not use_gpu:
            cmd.append("--cpu")
        if debug:
            cmd.append("--debug")

        print("\nRunning CDBNN processing with command:", ' '.join(cmd))

        # Run process with direct terminal access
        process = subprocess.run(
            cmd,
            stdin=None,  # Use the parent's stdin
            stdout=None, # Use the parent's stdout
            stderr=None, # Use the parent's stderr
            text=True
        )

        if process.returncode != 0:
            print(f"\nError in CDBNN processing - return code: {process.returncode}")
            return False

        return True

    except Exception as e:
        print(f"\nError running CDBNN: {str(e)}")
        if debug:
            traceback.print_exc()
        return False

def handle_command_line_args():
    """Parse and handle command line arguments"""
    parser = argparse.ArgumentParser(description='BDBNN Pipeline',
                                   usage='%(prog)s [options]',
                                   add_help=False)

    # Help argument
    parser.add_argument('-h', '--help', action='store_true',
                       help='Show this help message and exit')

    # Dataset options
    parser.add_argument('--data_type', choices=['torchvision', 'custom'],
                       help='Type of dataset')
    parser.add_argument('--data', help='Dataset name/path')
    parser.add_argument('--no-split', action='store_true',
                       help='Don\'t create train/test split')

    # CNN options
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--input-size', default='224,224',
                       help='Input image size (width,height)')
    parser.add_argument('--feature-dims', type=int, default=128,
                       help='Feature dimensions')

    # DBNN options
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of DBNN trials')
    parser.add_argument('--dbnn-epochs', type=int, default=1000,
                       help='DBNN training epochs')
    parser.add_argument('--dbnn-lr', type=float, default=0.1,
                       help='DBNN learning rate')
    parser.add_argument('--model-type', choices=['Histogram', 'Gaussian'],
                       default='Histogram', help='DBNN model type')

    # Execution options
    parser.add_argument('--cpu', '--no-gpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--skip-visuals', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--output-dir', default='data',
                       help='Output directory')
    parser.add_argument('--fresh', action='store_true',
                       help='Start fresh, ignore existing model')
    parser.add_argument('--config', help='Use existing configuration file')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Don\'t prompt for configuration editing')

    # Advanced options
    aug_group = parser.add_mutually_exclusive_group()
    aug_group.add_argument('--augment', action='store_true',
                          help='Enable data augmentation')
    aug_group.add_argument('--no-augment', action='store_true',
                          help='Disable data augmentation')
    parser.add_argument('--merge-datasets', action='store_true',
                       help='Merge train and test datasets')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')

    args = parser.parse_args()

    # Add default flag for GPU usage (inverse of --cpu)
    args.use_gpu = not args.cpu

    return args
def print_usage():
    """Print usage information with color-coded examples"""
    print(f"\n{Colors.BOLD}BDBNN (Bayesian Deep Neural Network) Pipeline{Colors.ENDC}")
    print("=" * 80)
    print("\nUsage:")
    print("  1. Interactive Mode:")
    print("     python bdbnn.py")

    print("\n  2. Command Line Mode:")
    print("     python bdbnn.py [options]")

    print(f"\n{Colors.BOLD}Options:{Colors.ENDC}")
    print("  Dataset Configuration:")
    print("    --data_type         Type of dataset ('torchvision' or 'custom')")
    print("    --data             Dataset name (torchvision) or path (custom)")
    print("    --no-split         Don't create train/test split for custom datasets")

    print("\n  CNN Training Parameters:")
    print("    --batch-size       Training batch size (default: 32)")
    print("    --epochs          Number of training epochs (default: 20)")
    print("    --learning-rate    Initial learning rate (default: 0.001)")
    print("    --workers         Number of data loading workers (default: 4)")
    print("    --input-size      Input image size (default: 224,224)")
    print("    --feature-dims    Feature dimensions (default: 128)")

    print("\n  DBNN Parameters:")
    print("    --trials          Number of DBNN trials (default: 100)")
    print("    --dbnn-epochs     DBNN training epochs (default: 1000)")
    print("    --dbnn-lr         DBNN learning rate (default: 0.1)")
    print("    --model-type      DBNN model type ('Histogram' or 'Gaussian')")

    print("\n  Execution Options:")
    print("    --cpu             Force CPU usage")
    print("    --no-gpu          Same as --cpu")
    print("    --debug           Enable debug mode")
    print("    --skip-visuals    Skip visualization generation")
    print("    --output-dir      Output directory (default: data)")
    print("    --fresh           Start fresh, ignore existing model")
    print("    --config          Use existing configuration file")
    print("    --no-interactive  Don't prompt for configuration editing")

    print("\n  Advanced Options:")
    print("    --augment         Enable data augmentation")
    print("    --no-augment      Disable data augmentation")
    print("    --merge-datasets  Merge train and test datasets")
    print("    --val-split       Validation split ratio (default: 0.2)")

    print(f"\n{Colors.BOLD}Examples:{Colors.ENDC}")
    print("  1. Basic interactive mode:")
    print("     python bdbnn.py")

    print("\n  2. Process custom dataset with specific parameters:")
    print("     python bdbnn.py --data_type custom --data path/to/images \\")
    print("                    --batch-size 64 --epochs 30 --learning-rate 0.0001")

    print("\n  3. Use torchvision dataset:")
    print("     python bdbnn.py --data_type torchvision --data MNIST")

    print("\n  4. Advanced configuration:")
    print("     python bdbnn.py --data_type custom --data path/to/images \\")
    print("                    --model-type Gaussian --trials 200 --dbnn-epochs 2000 \\")
    print("                    --input-size 299,299 --feature-dims 256")

# Add to bdbnn.py

def run_adbnn_process(dataset_name: str, config_path: str, use_gpu: bool = True, debug: bool = False) -> bool:
    """Run ADBNN process with proper arguments

    Args:
        dataset_name: Name of the dataset
        config_path: Path to configuration file
        use_gpu: Whether to use GPU
        debug: Whether to enable debug mode

    Returns:
        bool: True if process succeeded, False otherwise
    """
    try:
        # Build command line arguments
        cmd = [
            "python",
            "adbnn.py",
            "--dataset", dataset_name,
            "--config", config_path
        ]

        # Add GPU flag if needed
        if use_gpu and torch.cuda.is_available():
            cmd.append("--use_gpu")
        else:
            cmd.append("--cpu")

        # Add debug flag
        if debug:
            cmd.append("--debug")

        # Run ADBNN with real-time output
        print(f"\nRunning ADBNN processing with command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Process and display output in real-time
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
                output_lines.append(line)

        return_code = process.wait()

        if return_code != 0:
            print(f"\nError in ADBNN processing - return code: {return_code}")
            return False

        return True

    except Exception as e:
        print(f"\nError running ADBNN: {str(e)}")
        if debug:
            traceback.print_exc()
        return False

# Then modify the run_processing_pipeline function:
def run_processing_pipeline(dataset_name: str, dataset_path: str, dataset_type: str,
                          cnn_config_path: str, dbnn_config_path: str, args):
    """Run the complete processing pipeline with proper cleanup"""
    import gc
    import os
    import torch
    import traceback
    try:
        # Clean up before starting pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force close any lingering file handles
        try:
            import resource
            import gc

            # Force garbage collection
            gc.collect()

            # Get all open file descriptors
            import psutil
            process = psutil.Process()
            for handler in process.open_files():
                try:
                    os.close(handler.fd)
                except:
                    pass

        except ImportError:
            pass  # psutil not available

        # Step 1: Run CDBNN processing
        print("\nStep 1: Running CDBNN processing...")
        if not run_cdbnn_process(
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            config_path=cnn_config_path,
            use_gpu=not getattr(args, 'cpu', False),
            debug=getattr(args, 'debug', False)
        ):
            raise Exception("CDBNN processing failed")

        # Clean up after CDBNN
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 2: Run DBNN processing
        print("\nStep 2: Running DBNN processing...")
        if not run_adbnn_process(
            dataset_name=dataset_name,
            config_path=dbnn_config_path,
            use_gpu=not getattr(args, 'cpu', False),
            debug=getattr(args, 'debug', False)
        ):
            raise Exception("ADBNN processing failed")

        # Rest of the visualization code...
        return True

    except Exception as e:
        print(f"Error in processing pipeline: {str(e)}")
        if getattr(args, 'debug', False):
            traceback.print_exc()
        return False
    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Close any remaining file handles
        try:
            import psutil
            process = psutil.Process()
            for handler in process.open_files():
                try:
                    os.close(handler.fd)
                except:
                    pass
        except ImportError:
            pass

class DatasetHandler:
    """Handles dataset loading and property detection for various input types"""

    def __init__(self, base_data_dir: str = 'data'):
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Map of supported torchvision datasets and their properties
        self.torchvision_datasets = {
            'MNIST': {
                'class': torchvision.datasets.MNIST,
                'channels': 1,
                'size': [28, 28],
                'mean': [0.1307],
                'std': [0.3081]
            },
            'CIFAR10': {
                'class': torchvision.datasets.CIFAR10,
                'channels': 3,
                'size': [32, 32],
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2470, 0.2435, 0.2616]
            },
            'CIFAR100': {
                'class': torchvision.datasets.CIFAR100,
                'channels': 3,
                'size': [32, 32],
                'mean': [0.5071, 0.4867, 0.4408],
                'std': [0.2675, 0.2565, 0.2761]
            }
        }

    def _extract_archive(self, archive_path: str, extract_dir: Path) -> Path:
        """
        Extract compressed archives (zip, tar, etc)

        Args:
            archive_path: Path to the archive file
            extract_dir: Directory to extract to

        Returns:
            Path: Path to the root directory containing images
        """
        archive_path = Path(archive_path)

        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")

        # Find the root directory containing images
        image_dirs = []
        for root, dirs, files in os.walk(extract_dir):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) for f in files):
                image_dirs.append(root)

        if not image_dirs:
            raise ValueError("No image files found in archive")

        # Use the most shallow directory containing images
        return Path(min(image_dirs, key=lambda x: len(Path(x).parts)))

    def _setup_dataset_directory(self, input_path: str) -> Tuple[Path, str]:
        """
        Setup dataset directory structure and determine dataset name

        Args:
            input_path: Path to the input dataset

        Returns:
            Tuple[Path, str]: (data root directory, dataset name)
        """
        input_path = Path(input_path)

        # Handle different input types
        if input_path.is_file():
            if input_path.suffix.lower() in ['.zip', '.tar', '.gz', '.tgz']:
                # Compressed archive
                dataset_name = input_path.stem
                dataset_dir = self.base_data_dir / dataset_name
                if not dataset_dir.exists():
                    dataset_dir.mkdir(parents=True)
                    data_root = self._extract_archive(input_path, dataset_dir)
                else:
                    data_root = next(dataset_dir.glob("*"))  # First subdirectory
            else:
                raise ValueError(f"Unsupported file type: {input_path}")
        elif input_path.is_dir():
            # Directory containing images
            dataset_name = input_path.name
            dataset_dir = self.base_data_dir / dataset_name
            if not dataset_dir.exists():
                shutil.copytree(input_path, dataset_dir)
            data_root = dataset_dir
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

        return data_root, dataset_name

    def generate_default_config(self, folder_path: str) -> Dict:
        """
        Generate configuration files with proper cleanup

        Args:
            folder_path: Path to dataset folder

        Returns:
            Dict containing three configurations: json_config, dbnn_config, and data_config
        """
        try:
            # Normalize path and get dataset name
            folder_path = os.path.abspath(folder_path)
            dataset_name = os.path.basename(os.path.normpath(folder_path))

            # Find first image for properties detection
            first_image = None
            image_files = []
            for root, _, files in os.walk(folder_path):
                for f in files:
                    if any(f.lower().endswith(ext) for ext in self.SUPPORTED_IMAGE_EXTENSIONS):
                        image_files.append(os.path.join(root, f))
                        if not first_image:
                            first_image = os.path.join(root, f)
                        break
                if first_image:
                    break

            if not first_image:
                raise ValueError(f"No valid images found in {folder_path}")

            # Get image properties with proper cleanup
            channels, input_size = self._analyze_image_properties(first_image)
            mean, std = self._compute_dataset_stats(Path(folder_path), channels)

            # Count classes from directory structure
            train_dir = os.path.join(folder_path, 'train')
            if os.path.exists(train_dir):
                class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            else:
                class_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            num_classes = len(class_dirs)

            if num_classes == 0:
                raise ValueError(f"No class directories found in {folder_path}")

            # 1. Generate main JSON config
            json_config = {
                "dataset": {
                    "_comment": "Dataset configuration settings",
                    "name": dataset_name,
                    "_name_comment": "Dataset name",
                    "type": "torchvision" if self.datatype.lower() == "torchvision" else "custom",
                    "_type_comment": "custom or torchvision",
                    "in_channels": channels,
                    "_channels_comment": "Number of input channels",
                    "num_classes": num_classes,
                    "_classes_comment": "Number of classes",
                    "input_size": list(input_size),
                    "_size_comment": "Input image dimensions",
                    "mean": mean,
                    "_mean_comment": "Normalization mean",
                    "std": std,
                    "_std_comment": "Normalization standard deviation",
                    "train_dir": train_dir,
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

            # 2. Generate DBNN config
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

            # 3. Generate data config
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

            return {
                "json_config": json_config,
                "dbnn_config": dbnn_config,
                "data_config": data_config
            }

        except Exception as e:
            logger.error(f"Error generating configuration: {str(e)}")
            raise

        finally:
            # Clean up image lists and force garbage collection
            del image_files
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _analyze_image_properties(self, image_path: Path) -> Tuple[int, List[int]]:
        """Analyze image to determine channels and size"""
        try:
            with Image.open(image_path) as img:
                if img.mode == 'L':
                    channels = 1
                elif img.mode == 'RGB':
                    channels = 3
                elif img.mode == 'RGBA':
                    channels = 4
                else:
                    channels = 1  # Default to grayscale

                size = list(img.size)
                return channels, size
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            raise

    def _compute_dataset_stats(self, data_root: Path, channels: int) -> Tuple[List[float], List[float]]:
        """Compute dataset statistics with proper file cleanup"""
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.gif']:
            image_files.extend(data_root.rglob(f'*{ext}'))

        if not image_files:
            return [0.5] * channels, [0.5] * channels

        sample_size = min(1000, len(image_files))
        sampled_files = np.random.choice(image_files, sample_size, replace=False)

        means = []
        stds = []

        for img_path in sampled_files:
            try:
                with Image.open(img_path) as img:
                    if img.mode != ('L' if channels == 1 else 'RGB'):
                        img = img.convert('L' if channels == 1 else 'RGB')
                    img_array = np.array(img) / 255.0
                    if channels == 1:
                        img_array = img_array[..., np.newaxis]
                    means.append(img_array.mean(axis=(0, 1)))
                    stds.append(img_array.std(axis=(0, 1)))
                    del img_array  # Explicitly delete large arrays
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {str(e)}")
                continue

        # Compute overall statistics
        mean = np.mean(means, axis=0).tolist()
        std = np.mean(stds, axis=0).tolist()

        # Cleanup
        del means
        del stds

        return mean, std



    def _get_torchvision_dataset(self, dataset_name: str) -> Dict:
        """
        Get or download torchvision dataset and return its properties

        Args:
            dataset_name: Name of the torchvision dataset (e.g., 'MNIST', 'CIFAR10')

        Returns:
            Dict containing dataset properties
        """
        dataset_name = dataset_name.upper()
        if dataset_name not in self.torchvision_datasets:
            raise ValueError(f"Unsupported torchvision dataset: {dataset_name}")

        dataset_info = self.torchvision_datasets[dataset_name]
        dataset_dir = self.base_data_dir / dataset_name

        print(f"\nChecking for {dataset_name} dataset...")

        try:
            # Try to download/load training set to get properties
            train_dataset = dataset_info['class'](
                root=str(self.base_data_dir),
                train=True,
                download=True
            )

            # Try to download/load test set
            test_dataset = dataset_info['class'](
                root=str(self.base_data_dir),
                train=False,
                download=True
            )

            num_classes = len(train_dataset.classes)
            classes = train_dataset.classes

            # Setup train/test directories if needed
            train_dir = dataset_dir / 'train'
            test_dir = dataset_dir / 'test'
            train_dir.mkdir(parents=True, exist_ok=True)
            test_dir.mkdir(parents=True, exist_ok=True)

            return {
                'name': dataset_name,
                'type': 'torchvision',
                'in_channels': dataset_info['channels'],
                'num_classes': num_classes,
                'input_size': dataset_info['size'],
                'mean': dataset_info['mean'],
                'std': dataset_info['std'],
                'train_dir': str(train_dir),
                'test_dir': str(test_dir),
                'classes': classes,
                'data_root': str(dataset_dir)
            }

        except Exception as e:
            self.logger.error(f"Error downloading/loading {dataset_name}: {str(e)}")
            raise

    def analyze_dataset(self, input_path: str) -> Dict:
        """
        Analyze dataset and return its properties

        Args:
            input_path: Path to dataset (directory, zip, tar file) or torchvision dataset name

        Returns:
            Dict containing dataset properties
        """
        # Check if this is a torchvision dataset
        if isinstance(input_path, str) and input_path in self.torchvision_datasets:
            return self._get_torchvision_dataset(input_path)

        # Setup directory structure for local dataset
        data_root, dataset_name = self._setup_dataset_directory(input_path)

        # Detect classes (subdirectories)
        classes = []
        if data_root.is_dir():
            classes = [d.name for d in data_root.iterdir() if d.is_dir()]

        if not classes:
            # No subdirectories - check for image files directly
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.gif']:
                image_files.extend(data_root.rglob(f'*{ext}'))
            if not image_files:
                raise ValueError(f"No image files found in {data_root}")
            # Single class dataset
            classes = ['default']

        # Analyze first image for basic properties
        first_image = None
        for class_dir in classes:
            if class_dir == 'default':
                search_dir = data_root
            else:
                search_dir = data_root / class_dir

            for ext in ['.jpg', '.jpeg', '.png', '.gif']:
                images = list(search_dir.rglob(f'*{ext}'))
                if images:
                    first_image = images[0]
                    break
            if first_image:
                break

        if not first_image:
            raise ValueError("No valid image files found")

        # Get image properties
        channels, input_size = self._analyze_image_properties(first_image)

        # Compute dataset statistics
        mean, std = self._compute_dataset_stats(data_root, channels)

        # Prepare train/test directories
        train_dir = self.base_data_dir / dataset_name / 'train'
        test_dir = self.base_data_dir / dataset_name / 'test'
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Return dataset properties
        return {
            'name': dataset_name,
            'type': 'custom',
            'in_channels': channels,
            'num_classes': len(classes),
            'input_size': input_size,
            'mean': mean,
            'std': std,
            'train_dir': str(train_dir),
            'test_dir': str(test_dir),
            'classes': classes,
            'data_root': str(data_root)
        }

    def analyze_dataset(self, input_path: str) -> Dict:
        """
        Analyze dataset and return its properties

        Args:
            input_path: Path to dataset (directory, zip, or tar file)

        Returns:
            Dict containing dataset properties
        """
        # Setup directory structure
        data_root, dataset_name = self._setup_dataset_directory(input_path)

        # Detect classes (subdirectories)
        classes = []
        if data_root.is_dir():
            classes = [d.name for d in data_root.iterdir() if d.is_dir()]

        if not classes:
            # No subdirectories - check for image files directly
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.gif']:
                image_files.extend(data_root.rglob(f'*{ext}'))
            if not image_files:
                raise ValueError(f"No image files found in {data_root}")
            # Single class dataset
            classes = ['default']

        # Analyze first image for basic properties
        first_image = None
        for class_dir in classes:
            if class_dir == 'default':
                search_dir = data_root
            else:
                search_dir = data_root / class_dir

            for ext in ['.jpg', '.jpeg', '.png', '.gif']:
                images = list(search_dir.rglob(f'*{ext}'))
                if images:
                    first_image = images[0]
                    break
            if first_image:
                break

        if not first_image:
            raise ValueError("No valid image files found")

        # Get image properties
        channels, input_size = self._analyze_image_properties(first_image)

        # Compute dataset statistics
        mean, std = self._compute_dataset_stats(data_root, channels)

        # Prepare train/test directories
        train_dir = self.base_data_dir / dataset_name / 'train'
        test_dir = self.base_data_dir / dataset_name / 'test'
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Return dataset properties
        return {
            'name': dataset_name,
            'type': 'custom',
            'in_channels': channels,
            'num_classes': len(classes),
            'input_size': input_size,
            'mean': mean,
            'std': std,
            'train_dir': str(train_dir),
            'test_dir': str(test_dir),
            'classes': classes,
            'data_root': str(data_root)
        }





def main():
    """Main execution function with dynamic dataset handling"""
    args = handle_command_line_args()

    if args.help:
        print_usage()
        return 0

    try:
        print("=== BDBNN Integrated Pipeline ===")
        print("This pipeline will:")
        print("1. Process your image dataset through CDBNN to create features")
        print("2. Create and configure necessary configuration files")
        print("3. Process through ADBNN for classification")
        print("4. Generate visualizations and diagnostics")

        # Get dataset information
        if args.data and args.data_type:
            dataset_path = args.data
            dataset_type = args.data_type
        else:
            dataset_path, dataset_type = get_dataset_info()

        # Keep original path case
        dataset_name = dataset_path if dataset_type == 'torchvision' else os.path.basename(os.path.normpath(dataset_path))

        # Initialize DatasetProcessor
        processor = DatasetProcessor(
            datafile=dataset_path,
            datatype=dataset_type,
            output_dir='data'
        )

        # Process dataset first
        print("\nProcessing dataset...")
        train_dir, test_dir = processor.process()

        # Check if we have both train and test directories and in Histogram mode
        merge_datasets = False
        if test_dir and (args.model_type == "Histogram" if args.model_type else True):
            merge_default = 'y'
            merge_response = input(f"\nFound both train and test directories. Merge them for training? (Y/n) [default: {merge_default}]: ").strip().lower()
            merge_datasets = merge_response in ['', 'y', 'yes'] if merge_default == 'y' else merge_response in ['y', 'yes']

        # Generate configuration
        config_dict = processor.generate_default_config(os.path.dirname(train_dir))
        config = config_dict["json_config"]

        # Update config if merging datasets
        if merge_datasets:
            config['training']['merge_train_test'] = True

        print("\nDataset processed:")
        print(f"Training directory: {train_dir}")
        print(f"Test directory: {test_dir}")
        print(f"Number of channels: {config['dataset']['in_channels']}")
        print(f"Number of classes: {config['dataset']['num_classes']}")
        print(f"Input size: {config['dataset']['input_size']}")
        if merge_datasets:
            print("Datasets will be merged for training")

        # Get DBNN parameters interactively if no args provided
        if not args.no_interactive:
            print("\nDBNN Parameter Configuration")
            print("-" * 40)

            model_type = input("\nSelect model type (Histogram/Gaussian) [default: Histogram]: ").strip()
            args.model_type = model_type if model_type in ["Histogram", "Gaussian"] else "Histogram"

            trials_input = input("\nNumber of trials [default: 100]: ").strip()
            args.trials = int(trials_input) if trials_input else 100

            epochs_input = input("Number of epochs [default: 1000]: ").strip()
            args.dbnn_epochs = int(epochs_input) if epochs_input else 1000

            lr_input = input("Learning rate [default: 0.1]: ").strip()
            args.dbnn_lr = float(lr_input) if lr_input else 0.1

            args.use_gpu = input("\nUse GPU if available? (y/n) [default: y]: ").lower().strip() != 'n'
            args.debug = input("Enable debug mode? (y/n) [default: n]: ").lower().strip() == 'y'
            args.skip_visuals = input("Skip visualization generation? (y/n) [default: n]: ").lower().strip() == 'y'

        # Create configurations
        print("\nCreating configurations...")
        try:
            # Save CNN config
            dataset_dir = Path('data') / dataset_name
            cnn_config_path = dataset_dir / f"{dataset_name}.json"
            dbnn_config_path = dataset_dir / "adaptive_dbnn.conf"

            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Generate DBNN config
            dbnn_config = {
                "training_params": {
                    "trials": args.trials if args and args.trials else 100,
                    "cardinality_threshold": 0.9,
                    "cardinality_tolerance": 4,
                    "learning_rate": args.dbnn_lr if args and args.dbnn_lr else 0.1,
                    "random_seed": 42,
                    "epochs": args.dbnn_epochs if args and args.dbnn_epochs else 1000,
                    "test_fraction": args.val_split if args and args.val_split else 0.2,
                    "enable_adaptive": True,
                    "modelType": args.model_type if args and args.model_type else "Histogram",
                    "compute_device": "auto",
                    "use_interactive_kbd": False,
                    "debug_enabled": args.debug if args and args.debug else False,
                    "Save_training_epochs": True,
                    "training_save_path": str(Path("training_data") / dataset_name),
                    "merge_datasets": merge_datasets
                },
                     "execution_flags": {
                    "train": True,
                    "train_only": False,
                    "predict": True,
                    "gen_samples": False,
                    "fresh_start": False,
                    "use_previous_model": True,
                }
            }

            # Save configs using context managers
            with open(cnn_config_path, 'w') as f:
                json.dump(config, f, indent=4)

            with open(dbnn_config_path, 'w') as f:
                json.dump(dbnn_config, f, indent=4)

            # Allow configuration editing
            if not args.no_interactive:
                edit = input("\nWould you like to edit the configuration files? (y/n): ").lower() == 'y'
                if edit:
                    if os.name == 'nt':  # Windows
                        os.system(f'notepad {cnn_config_path}')
                        os.system(f'notepad {dbnn_config_path}')
                    else:  # Unix-like
                        editor = os.environ.get('EDITOR', 'nano')
                        os.system(f'{editor} {cnn_config_path}')
                        os.system(f'{editor} {dbnn_config_path}')

                    # Re-read configs after editing
                    with open(cnn_config_path, 'r') as f:
                        config = json.load(f)
                    with open(dbnn_config_path, 'r') as f:
                        dbnn_config = json.load(f)

        except Exception as e:
            print(f"Error creating configurations: {str(e)}")
            if args.debug:
                traceback.print_exc()
            return 1

        # Run pipeline
        print("\nRunning processing pipeline...")
        success = run_processing_pipeline(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            cnn_config_path=str(cnn_config_path),
            dbnn_config_path=str(dbnn_config_path),
            args=args
        )

        if not success:
            raise Exception("Pipeline processing failed")

        print("\nPipeline execution completed successfully!")
        return 0

    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        if args.debug:
            print("\nFull error traceback:")
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
