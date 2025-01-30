import os
import sys
import json
import logging
import torch
import random
import pandas as pd
import numpy as np
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
from itertools import combinations

# Import the CDBNN and ADBNN modules
from cdbnn import CNNTrainer, DatasetProcessor, ConfigManager
from adbnn import DBNN, DatasetConfig

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

def create_default_configs(dataset_name: str, dataset_path: str, dataset_type: str, args=None) -> Tuple[str, str]:
    """Create default configuration files for CNN and DBNN if they don't exist

    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to dataset
        dataset_type: Type of dataset (torchvision/custom)
        args: Command line arguments (optional)
    """
    print(f"\nChecking configurations for {dataset_name}...")

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    # Create dataset-specific directory
    dataset_dir = os.path.join('data', dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Ensure train directory exists
    train_dir = os.path.join(dataset_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    # Define config file paths
    cnn_config_path = os.path.join('data', dataset_name, f"{dataset_name}.json")
    dbnn_config_path = os.path.join('data', dataset_name, "adaptive_dbnn.conf")
    data_config_path = os.path.join('data', dataset_name, f"{dataset_name}.conf")

    # Check if config files already exist
    configs_exist = os.path.exists(cnn_config_path) and os.path.exists(dbnn_config_path) and os.path.exists(data_config_path)

    if configs_exist:
        print(f"Found existing configuration files in {dataset_dir}")
        return cnn_config_path, dbnn_config_path

    print("Creating new configuration files...")

    # Get number of classes from directory structure
    if os.path.isdir(dataset_path):
        num_classes = len([d for d in os.listdir(dataset_path)
                          if os.path.isdir(os.path.join(dataset_path, d))])
    else:
        num_classes = 2  # Default if cannot determine

    # Parse input size from args or use default
    if args and args.input_size:
        try:
            width, height = map(int, args.input_size.split(','))
            input_size = [width, height]
        except:
            input_size = [224, 224]  # Default if parsing fails
    else:
        input_size = [224, 224]

    # Create CNN config
    cnn_config = {
        "_comment": "CNN configuration file",
        "dataset": {
            "_comment": "Dataset configuration",
            "name": dataset_name,
            "type": dataset_type,
            "_comment_type": "Use 'torchvision' for built-in datasets",
            "in_channels": 3,
            "num_classes": num_classes,
            "input_size": input_size,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "train_dir": train_dir,
            "test_dir": os.path.join(dataset_dir, 'test')
        },
        "model": {
            "_comment": "Model architecture settings",
            "feature_dims": args.feature_dims if args and args.feature_dims else 128,
            "learning_rate": args.learning_rate if args and args.learning_rate else 0.001
        },
        "training": {
            "_comment": "Training parameters",
            "batch_size": args.batch_size if args and args.batch_size else 32,
            "epochs": args.epochs if args and args.epochs else 20,
            "num_workers": args.workers if args and args.workers else min(4, os.cpu_count() or 1),
            "cnn_training": {
                "_comment": "CNN specific training settings",
                "resume": True,
                "fresh_start": args.fresh if args and args.fresh else False,
                "min_loss_threshold": 0.01,
                "checkpoint_dir": "Model/cnn_checkpoints"
            }
        },
        "execution_flags": {
            "_comment": "Execution control flags",
            "mode": "train_and_predict",
            "_comment_mode": "Options: train_and_predict, train_only, predict_only",
            "use_gpu": args.use_gpu if args and hasattr(args, 'use_gpu') else torch.cuda.is_available(),
            "fresh_start": args.fresh if args and args.fresh else False
        }
    }

    # Create DBNN config
    dbnn_config = {
        "_comment": "Configuration file for Adaptive DBNN training and execution",
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

            "_comment_2": "Model and computation settings",
            "modelType": args.model_type if args and args.model_type else "Histogram",
            "compute_device": "auto",
            "use_interactive_kbd": False,
            "debug_enabled": args.debug if args and args.debug else False,

            "_comment_3": "Training data management",
            "Save_training_epochs": True,
            "training_save_path": os.path.join("training_data", dataset_name)
        },
        "execution_flags": {
            "_comment": "Execution control flags",
            "train": True,
            "train_only": False,
            "predict": True,
            "gen_samples": False,
            "fresh_start": args.fresh if args and args.fresh else False,
            "use_previous_model": not (args and args.fresh)
        }
    }

    # Create data configuration
    data_config = {
        "_comment": "Main data configuration file",
        "file_path": os.path.join("data", dataset_name, f"{dataset_name}.csv"),
        "_comment_filepath": "Can also be a URL for remote datasets",
        "separator": ",",
        "has_header": True,
        "target_column": "target",

        "likelihood_config": {
            "_comment": "Settings for likelihood computation",
            "feature_group_size": 2,
            "max_combinations": 1000,
            "bin_sizes": [20],
            "_comment_bins": "Can be variable sizes for each feature, e.g. [20,33,64]"
        },

        "active_learning": {
            "_comment": "Active learning parameters",
            "tolerance": 1.0,
            "cardinality_threshold_percentile": 95,
            "strong_margin_threshold": 0.3,
            "marginal_margin_threshold": 0.1,
            "min_divergence": 0.1
        },

        "training_params": {
            "_comment": "Dataset-specific training parameters",
            "Save_training_epochs": True,
            "training_save_path": os.path.join("training_data", dataset_name)
        },

        "modelType": args.model_type if args and args.model_type else "Histogram"
    }

    # Save configurations only if they don't exist
    if not os.path.exists(cnn_config_path):
        with open(cnn_config_path, 'w') as f:
            json.dump(cnn_config, f, indent=4)
        print(f"Created CNN config: {cnn_config_path}")

    if not os.path.exists(dbnn_config_path):
        with open(dbnn_config_path, 'w') as f:
            json.dump(dbnn_config, f, indent=4)
        print(f"Created DBNN config: {dbnn_config_path}")

    if not os.path.exists(data_config_path):
        with open(data_config_path, 'w') as f:
            json.dump(data_config, f, indent=4)
        print(f"Created data config: {data_config_path}")

    return cnn_config_path, dbnn_config_path

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

def run_processing_pipeline(dataset_name: str, dataset_path: str, dataset_type: str,
                          cnn_config_path: str, dbnn_config_path: str, args=None):
    """Run the complete processing pipeline"""
    try:
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

        # Step 2: Run DBNN processing
        print("\nStep 2: Running DBNN processing...")
        from adbnn import  DBNN  # Import here to avoid circular imports
        dbnn = DBNN(dataset_name=dataset_name)
        results = dbnn.process_dataset(dbnn_config_path)

        # Step 3: Generate visualizations
        if not getattr(args, 'skip_visuals', False):
            print("\nStep 3: Generating visualizations...")
            viz_dir = Path(f'data/{dataset_name}/visualizations')
            viz_dir.mkdir(parents=True, exist_ok=True)

            features_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")
            if os.path.exists(features_path):
                data = pd.read_csv(features_path)
                # Call visualization functions here

        print(f"\nResults saved in: data/{dataset_name}/")
        return True

    except Exception as e:
        print(f"Error in processing pipeline: {str(e)}")
        if getattr(args, 'debug', False):
            traceback.print_exc()
        return False

def main():
    """Main execution function with fixed process calls"""
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
        if args.data_type and args.data:
            dataset_type = args.data_type
            dataset_path = args.data
        else:
            dataset_type = input("\nEnter dataset type (torchvision/custom): ").strip().lower()
            while dataset_type not in ['torchvision', 'custom']:
                print("Invalid type. Please enter 'torchvision' or 'custom'")
                dataset_type = input("Enter dataset type: ").strip().lower()

            dataset_path = input("Enter dataset path: ").strip()

        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        if dataset_type=='torchvision':
            dataset_name=dataset_name.upper()
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"Path: {dataset_path}")
        print(f"Type: {dataset_type}")

        # Get DBNN parameters interactively if no args provided
        if not args.no_interactive:
            print("\nDBNN Parameter Configuration")
            print("-" * 40)

            model_type = input("\nSelect model type (Histogram/Gaussian) [default: Histogram]: ").strip()
            args.model_type = model_type if model_type in ["Histogram", "Gaussian"] else "Histogram"

            args.trials = int(input("\nNumber of trials [default: 100]: ").strip() or "100")
            args.dbnn_epochs = int(input("Number of epochs [default: 1000]: ").strip() or "1000")
            args.dbnn_lr = float(input("Learning rate [default: 0.1]: ").strip() or "0.1")
            args.use_gpu = input("\nUse GPU if available? (y/n) [default: y]: ").lower().strip() != 'n'
            args.debug = input("Enable debug mode? (y/n) [default: n]: ").lower().strip() == 'y'
            args.skip_visuals = input("Skip visualization generation? (y/n) [default: n]: ").lower().strip() == 'y'

        # Create configurations
        print("\nCreating configurations...")
        cnn_config_path, dbnn_config_path = create_default_configs(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            args=args
        )

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

        # Run pipeline
        print("\nRunning processing pipeline...")
        success = run_processing_pipeline(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            cnn_config_path=cnn_config_path,
            dbnn_config_path=dbnn_config_path,
            args=args
        )

        if not success:
            raise Exception("Pipeline processing failed")

    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        if args.debug:
            print("\nFull error traceback:")
            traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
