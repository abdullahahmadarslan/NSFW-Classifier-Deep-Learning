
# NSFW Image Classifier using EfficientNet

This project aims to classify images as NSFW (Not Safe For Work) using a deep learning model based on the **EfficientNet** architecture. The project leverages the `datasets` library to load a dataset, applies data augmentation techniques, and trains a classifier for NSFW content detection.

## Project Overview
The project uses the **EfficientNet** model, which is known for its balance of accuracy and efficiency, as the core architecture for image classification. The training process includes preprocessing, data augmentation, and fine-tuning of a pre-trained model on a dataset specifically labeled for NSFW detection.

## Features
- Image preprocessing and augmentation (resizing, random flipping, color jitter, etc.).
- Fine-tuned **EfficientNet** model for NSFW classification.
- Classification of images into NSFW categories using a pretrained model.
- Evaluation of the model using accuracy and loss metrics.

## Dependencies

To install the necessary dependencies, run the following command:

```bash
pip install datasets transformers torch torchvision efficientnet_pytorch
```

## Dataset
The dataset used for this project is `deepghs/nsfw_detect` and is available via the `datasets` library.

To load the dataset:

```python
from datasets import load_dataset
dataset = load_dataset("deepghs/nsfw_detect")
```

### Class Distribution
The dataset contains different categories for NSFW content, which are loaded and checked for distribution:

```python
print(dataset['train'].features['label'].names)  # Check the labels
print(dataset['train'].unique('label'))  # Unique label values
```

## Data Preprocessing
The images undergo a series of transformations before being fed into the model, including:

- **Resizing** to 224x224 pixels.
- **RandomHorizontalFlip** to randomly flip images horizontally.
- **RandomRotation** for small rotations.
- **ColorJitter** to adjust brightness, contrast, saturation, and hue.
- **Normalization** using ImageNet statistics.

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])
```

## Model Architecture
The **EfficientNet** model is used for classification. EfficientNet is a family of models that scale efficiently both in terms of size and computation while maintaining high accuracy.

```python
from efficientnet_pytorch import EfficientNet

# Load a pre-trained EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b0')
```

## Training the Model
The training process includes applying the above transformations and feeding the dataset to the EfficientNet model. The model is fine-tuned using standard training loops with PyTorch, optimizing for accuracy and minimizing loss.

## Evaluation
The model is evaluated on the validation set using standard classification metrics, including **accuracy** and **loss**. Visualizations for loss curves and accuracy trends over time can be generated.

## Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nsfw-classifier.git
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python train.py
   ```

## Future Work
- Hyperparameter tuning for better performance.
- Experiment with different architectures like ResNet or Vision Transformers.
- Deploy the model in a real-world application (e.g., content filtering, moderation tools).

