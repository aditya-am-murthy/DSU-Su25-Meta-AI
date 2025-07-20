# Video Similarity Learning - Educational Project

This project simplifies the Video Similarity Challenge for educational purposes, making it accessible for students learning computer vision and deep learning.

## 🎯 Project Overview

This project teaches you how to build a video similarity detection system using deep learning. You'll learn to:
- Extract features from video frames using pre-trained neural networks
- Train models to identify similar videos
- Use modern deep learning architectures (ResNet, ViT, etc.)
- Implement similarity matching algorithms
- Log experiments with Weights & Biases (wandb)

## 📁 Project Structure

```
VideoSimilarityLearning/
├── data/                   # All datasets and processed data
├── notebooks/              # Jupyter notebooks for learning
├── models/                 # Model architectures
├── utils/                  # Utility functions
├── configs/                # Configuration files
├── scripts/                # Training and evaluation scripts
└── requirements.txt        # Python dependencies
```

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Create a new conda environment
conda create -n video_similarity python=3.9
conda activate video_similarity

# Install dependencies
pip install -r requirements.txt
```

**⚠️ If you encounter NumPy compatibility issues**, run the fix script:
```bash
python fix_dependencies.py
```

### 2. Download Data

The project uses a simplified version of the Video Similarity Challenge dataset. Run the data download script:

```bash
python scripts/download_data.py
```

### 3. Start Learning

Begin with the notebooks in order:

1. **01_Data_Exploration.ipynb** - Understand the dataset
2. **02_Feature_Extraction.ipynb** - Learn to extract features from videos
3. **03_Model_Architecture.ipynb** - Explore different model architectures
4. **04_Training_Setup.ipynb** - Set up training configuration
5. **05_Model_Training.ipynb** - Train your first model
6. **06_Evaluation.ipynb** - Evaluate model performance

## 📚 Learning Objectives

### Week 1: Data Understanding
- Video data structure and preprocessing
- Frame extraction and augmentation
- Dataset organization

### Week 2: Feature Extraction
- Understanding pre-trained models
- Feature extraction from video frames
- Dimensionality reduction

### Week 3: Model Architecture
- CNN vs Transformer architectures
- Similarity learning approaches
- Model customization

### Week 4: Training & Evaluation
- Training loop implementation
- Loss functions for similarity learning
- Evaluation metrics
- Experiment logging with wandb

## 🛠️ Key Technologies

- **PyTorch** - Deep learning framework
- **OpenCV** - Video processing
- **Weights & Biases** - Experiment tracking
- **Jupyter** - Interactive learning
- **NumPy/Pandas** - Data manipulation

## 📊 Expected Outcomes

By the end of this project, you will:
- Understand video similarity detection
- Be able to implement custom model architectures
- Know how to train and evaluate deep learning models
- Have experience with experiment tracking
- Be ready for more advanced computer vision projects

## 🤝 Contributing

This is an educational project. Feel free to:
- Improve the documentation
- Add new model architectures
- Create additional learning materials
- Report issues or suggest improvements

## 📄 License

This project is for educational purposes. The original Video Similarity Challenge is from CVPR 2023.

## 🙏 Acknowledgments

- Original VSC Challenge organizers
- PyTorch and the open-source community
- Weights & Biases for experiment tracking tools 