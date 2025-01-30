## Example
 python bdbnn.py 
``` 
=== BDBNN Integrated Pipeline ===

This pipeline will:
1. Process your image dataset through CDBNN to create features
2. Create and configure necessary configuration files
3. Process through ADBNN for classification
4. Generate visualizations and diagnostics

Enter dataset type (torchvision/custom): custom
Enter dataset path: RingGalaxies/galaxies


Processing dataset: galaxies
Path: RingGalaxies/galaxies
Type: custom

DBNN Parameter Configuration
----------------------------------------

Select model type (Histogram/Gaussian) [default: Histogram]: 

Number of trials [default: 100]: 
Number of epochs [default: 1000]: 
Learning rate [default: 0.1]: 

Use GPU if available? (y/n) [default: y]: 
Enable debug mode? (y/n) [default: n]: 
Skip visualization generation? (y/n) [default: n]: 

Creating configurations...

Checking configurations for galaxies...
Found existing configuration files in data/galaxies

Would you like to edit the configuration files? (y/n): y

Edit files...


Running processing pipeline...

Step 1: Running CDBNN processing...

Running CDBNN processing with command: python cdbnn.py --data_type custom --data RingGalaxies/galaxies --config data/galaxies/galaxies.json
2025-01-31 01:10:12,771 - INFO - Logging setup complete. Log file: logs/training_20250131_011012.log
2025-01-31 01:10:12,771 - INFO - Starting CNN training process...
2025-01-31 01:10:12,772 - INFO - Loading configuration from data/galaxies/galaxies.json
2025-01-31 01:10:12,772 - INFO - Processing dataset...
2025-01-31 01:10:12,783 - INFO - Found 2 class directories: ['NonRings', 'Rings']

BDBNN Pipeline User ManualCreate train/test split from class directories? (y/n): n

2025-01-31 01:45:07,855 - INFO - Created training directory structure in data/galaxies/train
2025-01-31 01:45:07,855 - INFO - Dataset processed: train_dir=data/galaxies/train, test_dir=None
2025-01-31 01:45:07,855 - INFO - Initializing CNN trainer...
2025-01-31 01:45:08,317 - INFO - Latest checkpoint: Model/cnn_checkpoints/galaxies_best.pth
2025-01-31 01:45:08,317 - INFO - Found previous checkpoint at Model/cnn_checkpoints/galaxies_best.pth
2025-01-31 01:45:08,410 - INFO - Checkpoint loaded successfully
2025-01-31 01:45:08,411 - INFO - Optimizer state loaded
2025-01-31 01:45:08,411 - INFO - Training history loaded
2025-01-31 01:45:08,411 - INFO - Successfully initialized model from checkpoint
2025-01-31 01:45:08,430 - INFO - Starting model training...
2025-01-31 01:45:14,097 - INFO - Epoch 20: Train [12632 samples] Loss 0.1529, Acc 94.21%                                     
2025-01-31 01:45:14,111 - INFO - Saved latest checkpoint to Model/cnn_checkpoints/galaxies_checkpoint.pth
2025-01-31 01:45:14,123 - INFO - Saved best checkpoint to Model/cnn_checkpoints/galaxies_best.pth
2025-01-31 01:45:14,123 - INFO - Extracting features...
Extracting features: 100%|██████████████████████████████████████████████████████████████| 395/395 [00:05<00:00, 74.08batch/s]
2025-01-31 01:45:20,779 - INFO - Saved features to data/galaxies/galaxies.csv
2025-01-31 01:45:20,780 - INFO - Features saved to data/galaxies/galaxies.csv
2025-01-31 01:45:20,996 - INFO - Training history plot saved to data/galaxies/training_history.png
2025-01-31 01:45:20,996 - INFO - Processing completed successfully!

Step 2: Running DBNN processing...

Running ADBNN processing with command: python adbnn.py --dataset galaxies --config data/galaxies/adaptive_dbnn.conf --use_gpu
DBNN Dataset Processor
========================================
Found dataset pair:
  Config: data/galaxies.conf
  Data  : data/galaxies/galaxies.csv

Found 1 dataset pair(s)

============================================================
Dataset: galaxies
Config file: data/galaxies.conf
Data file: data/galaxies/galaxies.csv
============================================================

Dataset Information:
Dataset name: galaxies
Configuration file: data/galaxies.conf (1.2 KB)
Data file: data/galaxies/galaxies.csv (17927.9 KB)
Model type: Not specified
Process this dataset? (y/n): Please enter 'y' or 'n' y

Using default data file: data/galaxies/galaxies.csv
Inferred column names from CSV: ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30', 'feature_31', 'feature_32', 'feature_33', 'feature_34', 'feature_35', 'feature_36', 'feature_37', 'feature_38', 'feature_39', 'feature_40', 'feature_41', 'feature_42', 'feature_43', 'feature_44', 'feature_45', 'feature_46', 'feature_47', 'feature_48', 'feature_49', 'feature_50', 'feature_51', 'feature_52', 'feature_53', 'feature_54', 'feature_55', 'feature_56', 'feature_57', 'feature_58', 'feature_59', 'feature_60', 'feature_61', 'feature_62', 'feature_63', 'feature_64', 'feature_65', 'feature_66', 'feature_67', 'feature_68', 'feature_69', 'feature_70', 'feature_71', 'feature_72', 'feature_73', 'feature_74', 'feature_75', 'feature_76', 'feature_77', 'feature_78', 'feature_79', 'feature_80', 'feature_81', 'feature_82', 'feature_83', 'feature_84', 'feature_85', 'feature_86', 'feature_87', 'feature_88', 'feature_89', 'feature_90', 'feature_91', 'feature_92', 'feature_93', 'feature_94', 'feature_95', 'feature_96', 'feature_97', 'feature_98', 'feature_99', 'feature_100', 'feature_101', 'feature_102', 'feature_103', 'feature_104', 'feature_105', 'feature_106', 'feature_107', 'feature_108', 'feature_109', 'feature_110', 'feature_111', 'feature_112', 'feature_113', 'feature_114', 'feature_115', 'feature_116', 'feature_117', 'feature_118', 'feature_119', 'feature_120', 'feature_121', 'feature_122', 'feature_123', 'feature_124', 'feature_125', 'feature_126', 'feature_127', 'target']
Using default data file: data/galaxies/galaxies.csv
Using default target column: 'target'

[DEBUG] ====== Starting preprocessing ======
Loading previous model state
No previous model found - starting fresh
[DEBUG] Weight initialization complete. Structure:
- Number of classes: 2
- Class 0: 1000 feature pairs
- Class 1: 1000 feature pairs

Round 1/1000
Training set size: 4
Test set size: 12628
Training time for epoch 1 is: 83.15 seconds

......

```
## Overview
The BDBNN Pipeline is an integrated system that combines CNN feature extraction with Bayesian probabilistic classification. It consists of three main components:

1. **CDBNN (Convolutional Deep Neural Network Interface)**
   - Handles image feature extraction
   - Processes both built-in torchvision datasets and custom image datasets
   - Creates standardized features and configurations

2. **ADBNN (Adaptive Difference Boosting Neural Network)**
   - Performs Bayesian probabilistic classification
   - Uses adaptive learning to improve classification accuracy
   - Handles both histogram and Gaussian models

3. **BDBNN (Bridge Module)**
   - Integrates CDBNN and ADBNN
   - Provides visualization and analysis tools
   - Manages configuration and file organization

## Directory Structure
```
data/
├── dataset_name/              # Dataset-specific directory
│   ├── dataset_name.csv      # Extracted features
│   ├── dataset_name_predictions.csv
│   └── visualizations/       # Visualization outputs
├── dataset_name.conf         # CNN configuration
└── adaptive_dbnn.conf        # DBNN configuration
```

## Usage Instructions

1. **Dataset Preparation**
   - For custom datasets:
     - Place images in `data/dataset_name/`
     - Organize in train/test subdirectories
     - Each class should have its own subdirectory
   - For torchvision datasets:
     - Just specify the dataset name (e.g., MNIST)

2. **Running the Pipeline**
   ```bash
   python bdbnn.py
   ```
   - Enter dataset type (torchvision/custom)
   - Provide dataset path or name
   - The pipeline will automatically:
     - Extract features using CDBNN
     - Allow configuration editing
     - Run classification using ADBNN
     - Generate visualizations

3. **Configuration Editing**
   - CNN configuration: Controls feature extraction parameters
   - DBNN configuration: Controls classification parameters
   - Both can be edited during pipeline execution

4. **Output Files**
   - Features CSV: Extracted image features
   - Predictions CSV: Classification results
   - Visualization files: Interactive plots and analyses
   - Results text file: Performance metrics

## Key Features

1. **Feature Extraction (CDBNN)**
   - Handles multiple image formats
   - Automatic data augmentation
   - GPU acceleration when available
   - Configurable CNN architecture

2. **Classification (ADBNN)**
   - Adaptive learning for improved accuracy
   - Two modeling options:
     - Histogram model for discrete data
     - Gaussian model for continuous data
   - Automatic parameter tuning

3. **Visualization (BDBNN)**
   - t-SNE plots for feature visualization
   - Confusion matrices
   - Training progress plots
   - Interactive HTML outputs

## Common Issues and Solutions

1. **Memory Issues**
   - Reduce batch size in CNN configuration
   - Use data subsampling for large datasets

2. **Performance Issues**
   - Enable GPU usage in configuration
   - Adjust feature dimensions

3. **Configuration Issues**
   - Check file permissions
   - Verify JSON syntax in config files

## Dependencies
- PyTorch
- NumPy
- Pandas
- Plotly
- scikit-learn
- matplotlib
- seaborn

## Best Practices

1. **Dataset Organization**
   - Use consistent naming conventions
   - Maintain clean class separation
   - Ensure balanced class distribution

2. **Configuration Management**
   - Back up working configurations
   - Document configuration changes
   - Use version control for configs

3. **Resource Management**
   - Monitor GPU memory usage
   - Use appropriate batch sizes
   - Clean up temporary files



# CDBNN
# CNN Deep Convolution  Network Interface (CDBNN) User Manual

## Overview
The CDBNN (CNN Deep Convolutional Network) is a hybrid deep learning framework that combines Convolutional Neural Networks (CNNs) with Deep Belief Networks. This implementation provides a flexible pipeline for feature extraction using CNNs and subsequent processing using DBNNs.

## Key Features
- Support for both built-in torchvision datasets and custom image datasets
- Flexible configuration system with JSON-based configuration files
- Automatic feature extraction using CNN
- Comprehensive data augmentation options
- Checkpoint saving and loading for interrupted training
- Progress tracking and logging
- Command-line interface for easy usage

## System Requirements
- Python 3.7+
- PyTorch
- torchvision
- CUDA-capable GPU (optional but recommended)
- Additional dependencies: tqdm, pandas, numpy, matplotlib, seaborn

## Directory Structure
```
project/
├── data/               # Dataset storage
├── Model/              # Model checkpoints
│   └── cnn_checkpoints/
├── Traininglog/       # Training logs and metrics
└── configs/           # Configuration files
```

## Components

### 1. DatasetProcessor
Handles dataset loading and preprocessing for both torchvision and custom datasets.

Key features:
- Automatic dataset download for torchvision datasets
- Support for compressed custom datasets (.zip, .tar, etc.)
- Data augmentation with configurable transforms
- Progress tracking during dataset processing

### 2. Feature Extractor CNN
Implements the CNN architecture for feature extraction.

Features:
- Configurable number of input channels and feature dimensions
- BatchNorm layers for stable training
- Adaptive pooling for variable input sizes

### 3. CNNTrainer
Manages the training process and feature extraction.

Capabilities:
- Checkpoint management
- Early stopping
- Progress logging
- Feature extraction and saving
- Training history visualization

## Usage

### 1. Command Line Interface
The script can be run in three ways:

a. Using a configuration file:
```bash
python cdbnn.py --config config.json
```

b. Using command line arguments:
```bash
python cdbnn.py --data_type torchvision --data MNIST --batch_size 32 --epochs 10
```

c. Interactive mode:
```bash
python cdbnn.py
```

Available command-line arguments:
```
Configuration:
  --config PATH          Path to configuration file
  --data_type TYPE      Dataset type (torchvision/custom)
  --data PATH           Dataset name or path

Training:
  --batch_size N        Training batch size
  --epochs N            Number of training epochs
  --workers N           Number of data loading workers
  --learning_rate LR    Learning rate

Execution:
  --output-dir DIR      Output directory
  --cpu                 Force CPU usage
  --debug              Enable debug mode
  --merge-datasets     Merge train and test datasets
```

### 2. Configuration File Format
The configuration file is in JSON format with the following structure:

```json
{
  "dataset": {
    "name": "MNIST",
    "type": "torchvision",
    "in_channels": 1,
    "num_classes": 10,
    "input_size": [28, 28],
    "mean": [0.5],
    "std": [0.5]
  },
  "model": {
    "architecture": "CNN",
    "feature_dims": 128,
    "learning_rate": 0.001
  },
  "training": {
    "batch_size": 32,
    "epochs": 20,
    "num_workers": 4
  },
  "execution_flags": {
    "mode": "train_and_predict",
    "use_gpu": true
  }
}
```

### 3. Custom Dataset Structure
Custom datasets should follow this structure:
```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image1.jpg
│       └── image2.jpg
└── test/
    ├── class1/
    │   └── image1.jpg
    └── class2/
        └── image1.jpg
```

## Training Process

1. **Dataset Loading**
   - Loads and preprocesses the dataset
   - Applies configured data augmentation
   - Creates data loaders for training

2. **Model Training**
   - Initializes the CNN feature extractor
   - Trains using the configured parameters
   - Saves checkpoints periodically
   - Monitors training progress

3. **Feature Extraction**
   - Extracts features using the trained CNN
   - Saves features to CSV files for DBNN processing
   - Generates necessary configuration files

## Output Files

1. **Training Logs**
   - `training_metrics.csv`: Detailed training metrics
   - Console output with progress information

2. **Model Files**
   - Checkpoint files in `Model/cnn_checkpoints/`
   - Best model saved separately

3. **Feature Files**
   - CSV files containing extracted features
   - Separate files for training and test sets
   - Configuration files for DBNN processing

## Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller input image sizes
   - Enable CPU-only mode with --cpu flag

2. **Slow Training**
   - Increase number of workers
   - Check disk I/O
   - Ensure GPU is being utilized

3. **Poor Performance**
   - Check learning rate
   - Verify dataset structure
   - Review data augmentation settings

## Best Practices

1. **Configuration**
   - Start with default configurations
   - Adjust parameters based on dataset size
   - Monitor validation metrics

2. **Dataset Preparation**
   - Ensure balanced class distribution
   - Verify image formats and sizes
   - Use appropriate augmentation

3. **Training**
   - Start with smaller epochs
   - Use checkpointing for large datasets
   - Monitor resource usage

## Support
For issues and questions:
1. Check the debug logs
2. Enable debug mode with --debug flag
3. Verify configuration files
4. Check system requirements

## References
- PyTorch documentation
- torchvision datasets
- Deep Belief Network literature
- python cdbnn.py --config config.json
- python cdbnn.py --data_type torchvision --data MNIST
- python cdbnn.py --data_type custom --data /path/to/dataset
- python cdbnn.py --data_type torchvision --data MNIST --batch_size 32 --epochs 10 --workers 4
- python cdbnn.py --data_type custom --data /path/to/dataset --batch_size 64 --epochs 20 --learning_rate 0.001

# ADBNN
# Adaptive DBNN (Deep Bayesian Neural Network) User Manual

## Overview
This implementation provides a GPU-optimized Deep Bayesian Neural Network with support for both histogram-based and Gaussian models. The system is designed for efficient processing of classification tasks with adaptive learning capabilities.

## Key Features
- GPU acceleration support
- Adaptive learning with dynamic sample selection
- Support for both histogram and Gaussian models
- Automatic handling of categorical variables
- Built-in cross-validation and model evaluation
- Support for continuing training from previous sessions
- Comprehensive metrics and visualization tools

## Directory Structure
```json
project/
├── data/
│   ├── adaptive_dbnn.conf     # Global DBNN config
│   ├── dataset_name.conf      # Dataset config
│   └── dataset_name/          # Dataset-specific folder
│       ├── dataset_name.csv   # Feature file (merged)
│       ├── dataset_name_train.csv  # Training features
│       └── dataset_name_test.csv   # Test features
├── Model/
│   └── cnn_checkpoints/
└── Traininglog/
```

## Configuration Files
Each dataset requires a configuration file (`.conf`) with the following structure:

```json
{
    "file_path": "path/to/data.csv",
    "column_names": ["feature1", "feature2", "target"],
    "target_column": "target",
    "modelType": "Histogram",  // or "Gaussian"
    "training_params": {
        "trials": 100,
        "cardinality_threshold": 0.9,
        "learning_rate": 0.1,
        "epochs": 1000,
        "test_fraction": 0.2,
        "enable_adaptive": true
    }
}
```

## Model Types

### Histogram Model
- Uses non-parametric binning approach
- Better for discrete or mixed-type features
- More memory intensive but can capture complex distributions
- Configuration: Set `modelType: "Histogram"` in config file

### Gaussian Model
- Uses parametric Gaussian mixture modeling
- Better for continuous features
- More computationally efficient
- Configuration: Set `modelType: "Gaussian"` in config file

## Usage

### Basic Usage
1. Place your dataset in the appropriate directory structure
2. Create a configuration file for your dataset
3. Run the main script:
```python
from adbnn import DBNN
model = DBNN(dataset_name="your_dataset")
results = model.process_dataset("path/to/config.conf")
```

### Advanced Usage

#### Adaptive Learning
```python
model = DBNN(dataset_name="your_dataset")
history = model.adaptive_fit_predict(max_rounds=10)
```

#### Prediction Only
```python
model = DBNN(dataset_name="your_dataset")
predictions = model.predict_and_save(save_path="predictions.csv")
```

## Key Parameters

### Training Parameters
- `learning_rate`: Learning rate for weight updates (default: 0.1)
- `max_epochs`: Maximum number of training epochs (default: 1000)
- `test_size`: Fraction of data for testing (default: 0.2)
- `batch_size`: Batch size for training (default: 32)

### Model Parameters
- `n_bins_per_dim`: Number of bins per dimension for histogram model
- `cardinality_threshold`: Threshold for feature cardinality filtering
- `cardinality_tolerance`: Precision for feature rounding

## Output Files

### Model Files
- `*_weights.json`: Model weights
- `*_encoders.json`: Categorical encoders
- `*_components.pkl`: Model components and parameters

### Results Files
- `*_predictions.csv`: Predictions with probabilities
- `*_confusion_matrix.png`: Confusion matrix visualization
- `*_probability_distributions.png`: Probability distribution plots
- `*_training_metrics.csv`: Training metrics history

## Performance Optimization

### GPU Usage
- The model automatically detects and uses available GPU
- Uses CUDA if available, falls back to CPU if not
- Optimized for batch processing on GPU

### Memory Management
- Automatic batch size optimization based on available memory
- Efficient tensor management for large datasets
- Caching of frequently used computations

## Troubleshooting

### Common Issues
1. **Memory Errors**
   - Reduce batch size
   - Enable GPU memory clearing
   - Use histogram model for smaller datasets

2. **Convergence Issues**
   - Adjust learning rate
   - Increase number of epochs
   - Check feature preprocessing

3. **Performance Issues**
   - Enable GPU acceleration
   - Optimize batch size
   - Use Gaussian model for large datasets

### Debug Mode
Enable debug logging in configuration:
```json
{
    "training_params": {
        "debug_enabled": true
    }
}
```

## Best Practices

1. **Data Preparation**
   - Clean and normalize data
   - Handle missing values
   - Check feature distributions

2. **Model Selection**
   - Use Histogram for complex distributions
   - Use Gaussian for continuous features
   - Consider memory vs speed tradeoffs

3. **Training**
   - Start with default parameters
   - Enable adaptive learning for complex datasets
   - Monitor training metrics

4. **Evaluation**
   - Use multiple metrics
   - Check confusion matrices
   - Analyze probability distributions

## Support
For more information or support:
- Check the code documentation
- Review the example configurations
- Check the debugging output
- Enable debug mode for detailed logging
  
