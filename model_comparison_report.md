# Model Comparison Report

## Objective
The goal of this project was to classify aerial images into Bird and Drone categories using deep learning models.

## Models Tested

### 1. CNN Model - Version 1
A custom CNN model with convolution, pooling, flatten, dense and dropout layers.

### 2. CNN Model - Version 2
An improved CNN architecture tested after modifying layers and structure.

### 3. Transfer Learning - MobileNetV2
Used pretrained MobileNetV2 with custom output layers.

## Comparison Table

| Model | Validation Accuracy | Parameters | Training Speed | Remarks |
|------|----------------------|-----------|---------------|--------|
| CNN V1 | ~79% | 11.1 Million | Slow | Better than second CNN |
| CNN V2 | ~74% | Lower than V1 | Moderate | Did not outperform V1 |
| MobileNetV2 | ~97% | 2.4 Million | Fastest | Best overall model |

## Final Model Selected

MobileNetV2 was selected because it achieved highest accuracy with fewer trainable parameters and faster training time.

## Learning Experience

By comparing three models, I understood that model complexity alone does not guarantee better performance. This comparison helped me understand why pretrained models are useful when dataset size is limited.
