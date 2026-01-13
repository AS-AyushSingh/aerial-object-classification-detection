# Model Comparison Report — Filled Example

Dataset: classification_dataset
Image size: 224x224
Batch size: 32

## Measured baseline — RandomForest (color histograms)
- Training time (s): 0.21
- Validation accuracy: 0.7443
- Test accuracy: 0.8140

## Simulated Custom CNN
- Training time (s) [simulated]: 0.64
- Best val accuracy [simulated]: 0.8143
- Test accuracy [simulated]: 0.8740

## Simulated Transfer Learning (MobileNetV2)
- Training time (s) [simulated]: 0.85
- Best val accuracy [simulated]: 0.8443
- Test accuracy [simulated]: 0.9140

## Notes
- RandomForest trained on simple color histogram features; results are a lower-bound baseline.
- Simulated improvements are illustrative: actual CNN/transfer results must be obtained by training real models.