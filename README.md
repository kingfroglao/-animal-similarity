---
title: Animal Similarity
emoji: 👁
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Animal Species Similarity Comparison

This project presents a system for comparing two animal images to determine whether they belong to the same species. It combines object detection, image classification, and semantic similarity techniques using a variety of deep learning models.

## Project Overview

We implemented a hybrid pipeline that includes:

- Object detection using DETR (facebook/detr-resnet-50)
- Species classification using ViT (google/vit-base-patch16-224)
- Embedding-based similarity comparison using:
  - MobileNetV2 (fine-tuned on Animals-10)
  - ResNet
  - CLIP
- Final decision by score-level fusion

This system is deployed as an interactive web demo on Hugging Face Spaces using Gradio.

## Dataset

The system is based on the Animals-10 dataset (from Kaggle), which contains over 28,000 images across 10 animal classes. We constructed image pairs to train and evaluate similarity models.

Constructed datasets:
- 95, 335, 635, and 935 image pairs
- Each pair is labeled as same species or different species

## Model Components

- **Object Detection**: DETR is used to highlight animal regions in the image.
- **Image Classification**: ViT is used to predict class labels for detected animals.
- **Feature Extraction**:
  - MobileNetV2 was fine-tuned using the Animals-10 dataset and used to extract compact embeddings.
  - ResNet50 and ViT are used as feature extractors.
  - CLIP is used to compute high-level semantic similarity between images.
- **Label Matching**: A simple method to check whether two predicted labels are identical.

## Fusion and Scoring

A weighted score is computed from selected models:

```
final_score = 0.6 * MobileNet + 0.3 * CLIP + 0.1 * LabelMatch
```

Based on this score, the system determines whether the animals are from the same species.

## Evaluation

We evaluated each model individually across datasets of increasing size. Metrics include accuracy, precision, recall, and F1 score.

Key findings:
- Fine-tuned MobileNetV2 achieved the best performance overall.
- CLIP also performed well, particularly in semantic-level similarity.
- Label matching was useful as a supporting signal.
- ViT and ResNet were less effective without fine-tuning.

## Demo Features

- Upload two animal images.
- System detects objects, classifies species, computes similarity, and makes a decision.
- The final decision is shown with a fusion score and species labels.

## Tech Stack

- PyTorch
- Hugging Face Transformers
- Gradio
- Matplotlib
- scikit-learn
- PIL
- Animals-10 dataset (Kaggle)

## Project Structure

```
├── app.py                        # Gradio interface
├── model_utils.py                # All models and helper functions
├── mobilenetv2_animals10_finetuned.pth  # Fine-tuned model checkpoint
├── requirements.txt
└── README.md
```