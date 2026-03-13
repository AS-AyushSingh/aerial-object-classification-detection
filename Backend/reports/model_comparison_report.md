# Model Comparison Report

This is a template to record model performance and comparison between the Custom CNN and Transfer Learning approaches.

## Experimental setup
- Dataset: `classification_dataset` (train/valid/test)
- Image size: 224x224
- Batch size: 32
- Augmentations: RandomFlip, RandomRotation(0.1), RandomZoom(0.1)

## Models
- Custom CNN: small ConvNet with BatchNorm, Dropout
- Transfer Learning: MobileNetV2 (imagenet) with global avg pooling and head dense layer

## Results
- Custom CNN
  - Training time: 
  - Best val accuracy: 
  - Test accuracy: 
  - Precision/Recall/F1: (see `results/classification_report.txt` for custom)
- Transfer Learning
  - Training time: 
  - Best val accuracy: 
  - Test accuracy: 
  - Precision/Recall/F1: (see `results_transfer/classification_report.txt`)

## Comparison
- Accuracy comparison: 
- Model size (MB): 
- Inference time (ms per image): 
- Notes on generalization / observed failure modes:

## Recommendations
- Best model to deploy: (name)
- Suggested improvements: more augmentation, class-balanced sampling, fine-tuning more base layers, or larger backbone.

