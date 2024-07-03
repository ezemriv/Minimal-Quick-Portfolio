---
layout: pages
title: "cifar10 classification"
---

# Neural Network Optimization and Evaluation for CIFAR10 dataset classification

## Introduction
In this project, various neural networks were optimized step by step to improve their performance on the CIFAR-10 dataset. The best-performing model was a ResNet34 implemented using FastAI, which achieved remarkable results with minimal code. This model underwent several phases of training and evaluation, including finding the optimal learning rate, fine-tuning, and applying Test Time Augmentation (TTA).

## ResNet34 (from FastAI)
In just a few lines, this FastAI code accomplishes a lot:
- Downloads the CIFAR-10 dataset.
- `ImageDataLoaders` or `DataBlocks` effortlessly creates data loaders for training and validation, applying image resizing, transformations, and normalization.
- `vision_learner` builds a pretrained ResNet34 model.
- `lr_find` automatically finds the optimal learning rate.
- `fine_tune` trains the model in two phases: first, with frozen pre-trained layers, and then with all layers unfrozen.

**Final accuracy in validation data: 0.9717**

```python
from fastai.vision.all import *

# Set up path and data
path = untar_data(URLs.CIFAR)
dls = ImageDataLoaders.from_folder(path, valid='test', item_tfms=Resize(224),
                                   batch_tfms=[*aug_transforms(size=224, min_scale=0.75),
                                               Normalize.from_stats(*imagenet_stats)])
dls.show_batch()

# Create learner
learn = vision_learner(dls, resnet34, metrics=[error_rate, accuracy])

# Find learning rate
learn.lr_find()

# Fine-tune
learn.fine_tune(4, freeze_epochs=1)

# Unfreeze and train with discriminative learning rates
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-6, 1e-4))

# Test Time Augmentation
preds, targs = learn.tta()
print(f"Final accuracy: {accuracy(preds, targs).item():.4f}")

# Save the model
learn.save('final_model')
```
