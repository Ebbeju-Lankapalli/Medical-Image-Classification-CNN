# Medical Image Classification CNN

A deep learning project that uses a custom Convolutional Neural Network (CNN) built with PyTorch to classify chest X-ray images into three categories: **COVID-19, NORMAL, and PNEUMONIA**.

## Problem Statement

Chest X-ray analysis is widely used for detecting lung diseases. However, manually analyzing large numbers of X-ray images can be time-consuming and requires expert radiologists.

This project demonstrates how CNNs can automatically learn visual patterns from chest X-ray images and classify them into different disease categories.

> **Medical Disclaimer:** This project is developed for educational and research purposes only. It is not a medical diagnostic system and should not be used as a replacement for qualified medical professionals.

## Project Objectives

- Automatically analyze chest X-ray images
- Detect disease-related visual patterns
- Classify X-ray images into COVID-19, NORMAL, and PNEUMONIA
- Build a complete CNN classification pipeline using PyTorch
- Train and validate the model
- Evaluate model performance using multiple metrics
- Test the trained model on unseen images
- Save the trained model for future inference

## Classification Goal

| Input | Output |
|---|---|
| 224 × 224 Grayscale Chest X-ray | COVID-19 / NORMAL / PNEUMONIA |

**Problem Type:** Multi-Class Image Classification

## Dataset

The dataset contains approximately **5,228 grayscale chest X-ray images** distributed across three classes.

| Class | Number of Images |
|---|---:|
| COVID-19 | 1,626 |
| NORMAL | 1,802 |
| PNEUMONIA | 1,800 |
| **Total** | **5,228** |

### Dataset Characteristics

- Image format: PNG
- Image type: Grayscale
- Original image size: 256 × 256
- Number of classes: 3
- Dataset is relatively balanced

## Project Workflow

    Chest X-Ray Images
            ↓
    Data Exploration
            ↓
    Data Loading
            ↓
    Train / Validation Split
            ↓
    Image Preprocessing
            ↓
    Custom PyTorch Dataset
            ↓
    DataLoader
            ↓
    CNN Model
            ↓
    Model Training
            ↓
    Validation
            ↓
    Best Model Checkpoint
            ↓
    Performance Evaluation
            ↓
    Unseen Image Testing
            ↓
    Prediction

## Image Preprocessing

Each X-ray image goes through the following preprocessing steps:

1. Convert image to grayscale
2. Resize image from 256 × 256 to 224 × 224
3. Convert image into a numerical array
4. Normalize pixel values from 0–255 to 0–1
5. Add the grayscale channel dimension
6. Convert the image into a PyTorch tensor

### Final Input Shape

    (1, 224, 224)

Where:

- `1` = grayscale channel
- `224` = image height
- `224` = image width

## Train / Validation Split

The dataset is divided using an **80/20 stratified split**.

- Training Set: 80%
- Validation Set: 20%

Stratification helps maintain a similar class distribution between the training and validation datasets.

## Custom PyTorch Dataset

A custom `XrayDataset` class is used to load and preprocess X-ray images dynamically.

Each sample contains:

    (image_tensor, label_tensor)

This allows images to be loaded efficiently during model training.

## DataLoader

PyTorch `DataLoader` is used to create batches for efficient training.

- Batch Size: 32
- Training Shuffle: Enabled
- Validation Shuffle: Disabled

## CNN Architecture

A custom Convolutional Neural Network is implemented using PyTorch.

    Input
    (1, 224, 224)
          ↓
    Conv2D: 1 → 32
          ↓
    ReLU
          ↓
    Conv2D: 32 → 32
          ↓
    ReLU
          ↓
    MaxPooling
          ↓
    Conv2D: 32 → 64
          ↓
    ReLU
          ↓
    Conv2D: 64 → 64
          ↓
    ReLU
          ↓
    MaxPooling
          ↓
    Conv2D: 64 → 128
          ↓
    ReLU
          ↓
    MaxPooling
          ↓
    Flatten
          ↓
    Fully Connected: 128
          ↓
    ReLU
          ↓
    Dropout: 0.5
          ↓
    Fully Connected: 3
          ↓
    Output

### Architecture Details

| Component | Configuration |
|---|---|
| Input Channels | 1 |
| Convolution Layers | 5 |
| Filters | 32 → 32 → 64 → 64 → 128 |
| Kernel Size | 3 × 3 |
| Activation Function | ReLU |
| Pooling | Max Pooling |
| Fully Connected Layer | 128 neurons |
| Dropout | 0.5 |
| Output Classes | 3 |

## Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Epochs | 16 |
| Input Size | 224 × 224 |
| Input Channels | 1 |
| Output Classes | 3 |
| Device | GPU if available, otherwise CPU |

## Loss Function

**CrossEntropyLoss** is used because this is a multi-class classification problem with three possible output classes.

## Optimizer

The model is trained using the **Adam optimizer** with a learning rate of `0.001`.

## Model Checkpointing

During training, the model is evaluated on the validation dataset after every epoch.

Whenever validation accuracy improves, the model weights are saved as the best checkpoint.

This allows the best-performing model to be used later instead of automatically using the final training epoch.

## Model Performance

The model was trained for **16 epochs**.

### Best Validation Performance

**Best Validation Accuracy: ~97.13%**

### Final Epoch Performance

- Training Accuracy: **99.57%**
- Validation Accuracy: **95.41%**

The difference between training and validation accuracy is an important observation when evaluating the model's generalization performance.

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training and validation curves
- Predictions on unseen images

## Classification Report

The validation evaluation achieved approximately:

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| COVID-19 | 0.97 | 0.99 | 0.98 |
| NORMAL | 0.92 | 0.99 | 0.95 |
| PNEUMONIA | 0.99 | 0.89 | 0.94 |

### Overall Performance

- **Accuracy:** ~95.41%
- **Macro F1-score:** ~0.95
- **Weighted F1-score:** ~0.95

The results show strong classification performance across all three classes. However, the recall for the PNEUMONIA class is lower than for COVID-19 and NORMAL.

## Confusion Matrix

A confusion matrix is used to analyze how the model's predictions are distributed across the three classes.

It helps identify:

- Correct predictions
- COVID-19 misclassifications
- NORMAL misclassifications
- PNEUMONIA misclassifications

## Training Curves

Training and validation curves are generated to monitor:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

These curves help analyze learning progress, convergence, and possible overfitting.

## Unseen Image Testing

After training and validation, the model is tested on previously unseen X-ray images.

The inference process is:

    Unseen X-Ray
         ↓
    Preprocessing
         ↓
    CNN Model
         ↓
    Class Scores
         ↓
    Predicted Class

Possible predictions:

- COVID-19
- NORMAL
- PNEUMONIA

## Model Saving

The trained model weights are saved using PyTorch.

Example model files:

- `best_model.pth`
- `xray_cnn_model.pth`

These model weights can be loaded later for inference without retraining the entire network.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow
- Scikit-learn
- Jupyter Notebook

## Project Structure

    Medical-Image-Classification-CNN/
    │
    ├── Medical_Image_Classification.ipynb
    ├── best_model.pth
    ├── xray_cnn_model.pth
    ├── requirements.txt
    └── README.md

The dataset is not included in the repository because of its size and medical-data considerations.

## How to Run

### 1. Clone the Repository

    git clone https://github.com/Ebbeju-Lankapalli/Medical-Image-Classification-CNN.git
    cd Medical-Image-Classification-CNN

### 2. Create a Conda Environment

    conda create -n medical-cnn python=3.11
    conda activate medical-cnn

### 3. Install Dependencies

    pip install -r requirements.txt

### 4. Launch Jupyter Notebook

    jupyter notebook

Open:

    Medical_Image_Classification.ipynb

Run the notebook cells sequentially.

## Requirements

The project requires the following Python libraries:

    numpy
    pandas
    matplotlib
    seaborn
    pillow
    torch
    scikit-learn
    jupyter

## Key Learnings

This project provided practical experience with:

- Medical image classification
- CNN architecture design
- PyTorch fundamentals
- Custom PyTorch Dataset creation
- DataLoader implementation
- Image preprocessing
- Multi-class classification
- CrossEntropyLoss
- Adam optimization
- GPU/CPU device handling
- Model checkpointing
- Training and validation monitoring
- Confusion matrix analysis
- Classification reports
- Model evaluation
- Unseen-data inference
- Saving and loading trained model weights

## Future Improvements

Potential improvements include:

- Data augmentation
- Batch normalization
- Learning-rate scheduling
- Early stopping
- Hyperparameter tuning
- Transfer learning
- ResNet-based models
- DenseNet-based models
- EfficientNet-based models
- ROC-AUC analysis
- Grad-CAM visualization
- Cross-validation
- External dataset evaluation

A useful next step would be comparing this custom CNN with pretrained architectures using transfer learning.

## Limitations

- This is an educational and research project.
- The dataset is relatively small compared with large-scale medical imaging datasets.
- Dataset characteristics may not represent all patient populations or imaging devices.
- High validation accuracy does not guarantee clinical reliability.
- The model has not been clinically validated.
- External validation would be required before considering real-world clinical use.
- The model should not be used to make medical decisions.

## Conclusion

This project demonstrates an end-to-end **CNN-based medical image classification pipeline using PyTorch**.

The system processes grayscale chest X-ray images and predicts one of three categories:

**COVID-19 | NORMAL | PNEUMONIA**

The model achieved approximately **95.41% validation accuracy**, with strong precision, recall, and F1-scores across the three classes.

The project demonstrates the complete workflow from image preprocessing and dataset preparation to CNN architecture design, model training, validation, evaluation, checkpointing, and inference on unseen images.

## Author

**Ebbeju Lankapalli**

GitHub: https://github.com/Ebbeju-Lankapalli

---

⭐ If you find this project useful, consider giving the repository a star!
