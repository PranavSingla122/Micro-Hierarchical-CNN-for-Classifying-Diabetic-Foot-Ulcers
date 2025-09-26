# Micro-Hierarchical CNN for Classifying Diabetic Foot Ulcers

A ultra-efficient deep learning system for diabetic foot ulcer detection and Wagner grading, achieving 95.87% binary accuracy and 90.63% multiclass accuracy with only 0.317 MB total model size.

## Overview

This project implements a micro-hierarchical CNN architecture that addresses critical barriers to clinical deployment of AI-based DFU assessment systems. The system features:

- **Binary Classification**: Detects ulcer presence/absence (95.87% accuracy)
- **Wagner Grading**: Classifies ulcer severity across grades 0-3 (90.63% accuracy) 
- **Ultra-Compact**: Complete system under 0.317 MB (99.67% parameter reduction vs ResNet)
- **Clinical-Ready**: Designed for Internet of Medical Things (IoMT) deployment

## Key Innovations

### Architectural Components
- **UltraEfficientConv**: Conditional depthwise separable convolutions (60-80% parameter reduction)
- **MicroAttention**: Channel attention with 8:1 reduction ratio (Cohen's d = 1.8 improvement)
- **Dual Pooling**: Combined global average and max pooling for enhanced feature representation
- **Hierarchical Design**: Two-stage system mimicking clinical decision workflow

### Training Enhancements
- **Enhanced Focal Loss**: Focal loss (γ=3.0) with label smoothing (α=0.2) for class imbalance
- **Strategic Oversampling**: 5× for minority classes (Grades 0,3), 3× for majority classes
- **8× Test-Time Augmentation**: Comprehensive ensemble with rotations, flips, and transformations
- **Differential Learning Rates**: Backbone (0.0005), attention (0.003), classifier (0.003)

## File Structure

```
micro-hierarchical-dfu-classifier/
├── patches1.py                           # Main implementation
├── confusion_matrix.png                  # Results visualization
├── README.md                            # Documentation
└── LICENSE                              # Apache 2.0 License
```

## Dataset

**Source**: [DFU Wagner's Classification Dataset](https://www.kaggle.com/datasets/purushomohan/dfu-wagners-classification)

**Composition** (1,495 total images):
- Normal skin: 606 images
- Grade 0: 65 images  
- Grade 1: 157 images
- Grade 2: 171 images
- Grade 3: 82 images
- False cases: 414 images (Burns, Bruises, Cuts, Abrasions)

## Model Architecture

### Binary Model (UltraMicroBinaryModel)
- **Size**: 15,121 parameters (0.058 MB)
- **Architecture**: 3-block progressive feature extraction
  - Initial: 3→8 channels with stride=2 downsampling
  - Block 1: 8→16 channels + MaxPool + MicroAttention
  - Block 2: 16→24 channels + MaxPool + MicroAttention  
  - Block 3: 24→32 channels + MaxPool + MicroAttention
  - Classifier: Global pooling + 2-layer FC with dropout

### Multiclass Model (UltraMicroMulticlassModel)  
- **Size**: 67,976 parameters (0.259 MB)
- **Architecture**: 4-stage backbone with dual pooling
  - Stem: 3→12 channels with aggressive downsampling
  - Stage 1: 12→20 channels + MicroAttention
  - Stage 2: 20→32 channels (2 layers) + MicroAttention
  - Stage 3: 32→48 channels (2 layers) + MicroAttention
  - Stage 4: 48→64 channels + MicroAttention
  - Feature Fusion: Dual pooling (avg + max) → 128D vector
  - Grade Attention: 128→32→4 channel-specific weighting
  - Classifier: 3-layer FC with batch normalization

## Performance Results

### Binary Classification
| Metric | Value |
|--------|-------|
| **Accuracy** | **95.87%** |
| Precision (Overall) | 0.910 |
| Recall (Overall) | 0.948 |
| F1-Score | 0.929 |
| AUC-ROC | 0.990 |

### Wagner Grade Classification (with 8× TTA)
| Grade | Precision | Recall | F1-Score | Clinical Significance |
|-------|-----------|--------|----------|----------------------|
| **Grade 0** | 0.91 | 0.98 | 0.94 | Superficial – monitoring |
| **Grade 1** | 0.95 | 0.91 | 0.93 | Partial thickness – care |
| **Grade 2** | 0.85 | 0.88 | 0.86 | Full thickness – immediate |
| **Grade 3** | 0.93 | 0.86 | 0.89 | Deep – urgent specialist |
| **Macro Avg** | **0.91** | **0.91** | **0.91** | **90.63% Overall Accuracy** |

### Robustness Evaluation
| Test Condition | Accuracy |
|----------------|----------|
| Bruises (False Positive) | 97.1% |
| Abrasions (False Positive) | 80.3% |
| Cuts (False Positive) | 79.4% |
| Burns (False Positive) | 71.9% |
| Adversarial (Noise/Blur) | 65-70% |

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/micro-hierarchical-dfu-classifier.git
cd micro-hierarchical-dfu-classifier

# Install dependencies  
pip install torch>=1.9.0 torchvision>=0.10.0
pip install scikit-learn>=0.24.0 matplotlib>=3.3.0 seaborn>=0.11.0
pip install pillow>=8.0.0 numpy>=1.20.0
```

## Training Configuration

### Binary Model Training
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (T_max=150, η_min=1e-6)  
- **Loss**: BCEWithLogitsLoss with positive weight adjustment
- **Batch Size**: 32
- **Epochs**: 150

### Multiclass Model Training  
- **Optimizer**: AdamW with component-specific learning rates
  - Backbone: 0.0005
  - Attention: 0.003  
  - Classifier: 0.003
- **Scheduler**: OneCycleLR with cosine annealing
- **Loss**: Enhanced Focal Loss (γ=3.0, α=0.2 label smoothing)
- **Augmentation**: Mixup (60% probability, β(0.3,0.3))
- **Batch Size**: 12  
- **Epochs**: 250

## Data Augmentation Strategies

### Binary Training (Standard)
- Resize: 256×256 → RandomCrop: 224×224
- Random rotation: ±30°
- Horizontal flip: 50% probability
- Color jitter: brightness/contrast/saturation=0.3, hue=0.1
- Random erasing: 15% probability

### Multiclass Training (Aggressive)
- Resize: 280×280 → RandomResizedCrop: 224×224 (scale 0.65-1.0)
- Random rotation: ±50°  
- Horizontal/vertical flips: 70%/50%
- Perspective distortion: 40% probability
- Gaussian blur: 40% probability
- Random erasing: 30% probability

## Ablation Study Results

| Architecture Variant | Size (MB) | Accuracy (%) | Effect Size |
|----------------------|-----------|--------------|-------------|
| **Full Architecture** | **0.014** | **96.5±1.2** | **-** |
| Standard Convolution | 0.049 | 94.2±0.5 | - |
| No Attention | 0.010 | 90.7±1.0 | Cohen's d = 1.8 |
| Single Pooling | 0.015 | 93.2±0.8 | Cohen's d = 0.6 |
| Deeper Channels | 0.041 | 95.0±0.6 | - |
| Baseline (DepthwiseSE) | 0.012 | 92.2±0.7 | - |

## Clinical Integration

### Wagner Classification Grades
- **Grade 0**: Superficial ulcer → Basic wound care and monitoring
- **Grade 1**: Partial thickness → Professional wound care required  
- **Grade 2**: Full thickness → Immediate medical attention needed
- **Grade 3**: Deep with bone involvement → Urgent specialist consultation

### IoMT Deployment Benefits
- **Edge Inference**: No cloud dependency, local processing
- **Privacy Compliant**: Patient data stays on device
- **Low Latency**: Real-time assessment in clinical workflow  
- **Resource Efficient**: Runs on mobile devices and embedded systems
- **Battery Friendly**: Minimal computational overhead

## Requirements

```txt
torch>=1.9.0
torchvision>=0.10.0  
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
Pillow>=8.0.0
numpy>=1.20.0
```

## Key Classes & Methods

### Core Components
- `UltraMicroClassifier`: Main hierarchical classification system
- `UltraMicroBinaryModel`: Ultra-efficient binary detection (15K params)
- `UltraMicroMulticlassModel`: Compact Wagner grading (68K params)
- `UltraEfficientConv`: Conditional depthwise separable convolution
- `MicroAttention`: Lightweight channel attention (8:1 reduction)
- `ImprovedFocalLoss`: Enhanced focal loss with label smoothing

### Training Pipeline
- `train_all_models()`: Complete hierarchical training workflow
- `prepare_binary_data()`: Strategic class balancing for binary task
- `prepare_multiclass_data()`: Oversampling for Wagner grade imbalance  
- `evaluate_models()`: Comprehensive evaluation with 8× TTA

### Clinical Interface
- `predict_hierarchical()`: Two-stage clinical assessment pipeline
- `test_single_image_ultra()`: Single image diagnostic testing

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [DFU Wagner's Classification Dataset](https://www.kaggle.com/datasets/purushomohan/dfu-wagners-classification) contributors
- Research institutions enabling healthcare AI development

---

**Medical Disclaimer**: This system is designed for research and educational purposes. Clinical decisions should always involve qualified healthcare professionals. The system provides diagnostic support but does not replace professional medical judgment.
