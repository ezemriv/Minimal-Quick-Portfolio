---
layout: pages
title: "cifar10 classification"
---

# CIFAR10 Image Classification CNN Optimization (final >97% in test)

In this project, I aim to systematically optimize a neural network classifier for this dataset, exploring both architectural modifications and the benefits of transfer learning.

## The CIFAR-10 Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

<img src="https://miro.medium.com/max/709/1*LyV7_xga4jUHdx4_jHk1PQ.png" 
        alt="Picture" 
        width="600" 
        height="400" 
        style="display: block; margin: 0 auto" />

## Initial Approach: Architecture Refinement

Initially, I focused on refining the model architecture itself. I've created the CIFAR10ModelTester class to streamline the evaluation process.

The goal in this phase was to determine the highest achievable test accuracy through iterative improvements to the architecture and hyperparameter tuning alone.

Techniques employed:

- **Dropout & Normalization**: Added dropout layers for regularization, batch normalization for stability.
- **Increased Complexity**: Gradually added convolutional layers, experimented with filter sizes.
- **Callbacks**: Implemented early stopping, model checkpointing, and learning rate scheduling.
- **Hyperparameter Tuning**: Tested batch sizes, optimized learning rate, weight decay.
- **Regularization**: Applied weight decay, utilized learning rate schedules.
- **Global Average Pooling**: Replaced fully connected layers to reduce overfitting.

### Best Test Accuracy (Architecture and Hyperparameter Tuning): **0.894**

## Transfer Learning Exploration

In the end, I explored the potential of transfer learning. I experimented with **Xception** and **EfficientNetB2** from Keras, as well as **ResNet34** from FastAI.

By fine-tuning these models on the CIFAR-10 dataset, I further boosted classification accuracy.

### Best Final Test Accuracy: **0.9706 using ResNet34**

<div class="button-container">
    <a href="https://github.com/ezemriv/CIFAR10_cnn_optimization" class="view-full-plot">View full code on GitHub</a>
  </div>

## ResNet34 Implementation (from fast.ai)

In just a few lines, this FastAI code accomplishes a lot:

- Downloads the CIFAR-10 dataset.
- ImageDataLoaders or Datablocks effortlessly creates data loaders for training and validation, applying image resizing, transformations, and normalization.
- vision_learner builds a pretrained ResNet34 model.
- lr_find automatically finds the optimal learning rate.
- fine_tune trains the model in two phases: first, with frozen pre-trained layers, and then with all layers unfrozen.

Final accuracy in validation data: **0.9717**

```python
from fastai.vision.all import *

# Set up path and data
path = untar_data(URLs.CIFAR)
dls = ImageDataLoaders.from_folder(path, valid='test', item_tfms=Resize(224),
                                   batch_tfms=[*aug_transforms(size=224, min_scale=0.75),
                                               Normalize.from_stats(*imagenet_stats)])
dls.show_batch()
```
![sample images](images\show_batch_fastai.png "images")
```python
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
```

## Custom Train, Val, Test Split

The CIFAR-10 dataset, in its standard format, is divided into 'train' and 'test' folders. FastAI's native ImageDataLoaders and DataBlocks, which are typically used to load such data, directly splits the data into training and validation sets.

This lacks the flexibility to create a separate test set (unseen for the model) from the original training data.

To address this, custom code was implemented to create a dedicated test set from the original training data. This involved shuffling the images, reserving 10,000 for test, and organizing them into temporary directories for model training and evaluation using FastAI's DataBlock.

### Final Results:

- Validation final accuracy: **0.9675**
- Evaluation accuracy on test images: **0.9600**
- Evaluation accuracy on test images with TTA: **0.9706**

### Code used for custom splitting, train and predict:

```python
from fastai.vision.all import *
import numpy as np
import shutil
from pathlib import Path

def setup_cifar_files(path, reserved_size=10000):
    # Get all training images
    train_path = path/'train'
    train_files = get_image_files(train_path)

    # Shuffle the files
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(train_files)

    # Extract reserved set
    reserved_files = train_files[:reserved_size]
    new_train_files = train_files[reserved_size:]

    # Create temporary directories
    temp_dir = Path('temp_cifar')
    temp_train_dir = temp_dir/'train'
    temp_test_dir = temp_dir/'test'
    temp_train_dir.mkdir(parents=True, exist_ok=True)
    temp_test_dir.mkdir(parents=True, exist_ok=True)

    # Copy new train files to temporary train directory with subdirectories
    for file in new_train_files:
        class_dir = file.parent.name
        dest_dir = temp_train_dir/class_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, dest_dir/file.name)

    # Copy test files to temporary test directory with subdirectories
    test_path = path/'test'
    for file in get_image_files(test_path):
        class_dir = file.parent.name
        dest_dir = temp_test_dir/class_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, dest_dir/file.name)

    return temp_dir, reserved_files

path = untar_data(URLs.CIFAR)
temp_dir, reserved_files = setup_cifar_files(path)

"""
Number of files in train: 40000
Number of files in test: 10000
Number of reserved files: 10000
"""

# Define the DataBlock
cifar_block = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=GrandparentSplitter(train_name='train', valid_name='test'),
    get_y=parent_label,
    item_tfms=Resize(224),
    batch_tfms=[*aug_transforms(size=224, min_scale=0.75),
       Normalize.from_stats(*imagenet_stats)]
)

dls = cifar_block.dataloaders(path, bs=64)  # Adjust batch size (bs) as needed

# Create learner
learn = vision_learner(dls, resnet34, metrics=[error_rate, accuracy])
# Fine-tune
learn.fine_tune(4, freeze_epochs=1)
# Unfreeze and train with discriminative learning rates
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-6, 1e-4))
# Test Time Augmentation
preds, targs = learn.tta()
print(f"Final accuracy: {accuracy(preds, targs).item():.4f}")
```

![custom train learner](images\learner_train.png "train")

```python
# Function to set up the evaluation directory with reserved samples
def setup_reserved_eval_dir(reserved_files, eval_dir):
    eval_dir.mkdir(parents=True, exist_ok=True)
    for file in reserved_files:
        class_dir = file.parent.name
        dest_dir = eval_dir/class_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, dest_dir/file.name)

# Create the evaluation directory
eval_dir = Path('temp_cifar_eval')
setup_reserved_eval_dir(reserved_files, eval_dir)

# Custom splitter to NOT split data
def nosplit(o):
    return L(int(i) for i in range(len(o))), L()

# Define the DataBlock for evaluation
cifar_block = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    item_tfms=Resize(224),
    batch_tfms=[*aug_transforms(size=224, min_scale=0.75),
       Normalize.from_stats(*imagenet_stats)],
    splitter=nosplit
)

# Create DataLoader for evaluation set
eval_dls = cifar_block.dataloaders(eval_dir, bs=64, shuffle_train=False, drop_last=False)

# Evaluate on all samples
eval_preds, eval_targs = learn.get_preds(dl=eval_dls.train)
eval_accuracy = accuracy(eval_preds, eval_targs).item()
print(f"Evaluation accuracy on all evaluation samples: {eval_accuracy:.4f}")

# Using test time augmentation
preds, targs = learn.tta(dl=eval_dls.train)
# Calculate accuracy
accuracy_score = accuracy(preds, targs).item()
print(f"Evaluation accuracy with TTA: {accuracy_score:.4f}")
```

<div class="button-container">
    <a href="https://github.com/ezemriv/CIFAR10_cnn_optimization" class="view-full-plot">View full code on GitHub</a>
  </div>