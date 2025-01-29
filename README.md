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
  
