# Structured Pruning for ResNet Model using CIFAR-10 Dataset

## Project Information

**Course:** SML (Statistical Machine Learning,) - Master Program
**University:** Hanoi University of Science and Technology (HUST)
**Project Duration:** 30/03/2026 - 08/04/2026

## Team Members

| Name | Student ID |
|------|-----------|
| Nguyen Tuan Anh | 20252752M |
| Dinh Quoc Dat | 20252745M |
| Ngo Quang Minh | 20252084M |
| Le Van Luan |20252572M |
| Nguyen Manh Hung | 20252276M |

## Project Overview

This project focuses on implementing **structured pruning techniques** for ResNet models trained on the CIFAR-10 dataset. Structured pruning aims to reduce model complexity by removing entire channels or filters from neural networks while maintaining performance, resulting in more efficient models suitable for deployment on resource-constrained devices.

## Key Objectives

- Implement structured pruning methods for ResNet architectures
- Evaluate pruning strategies on CIFAR-10 classification task
- Achieve model compression while maintaining classification accuracy
- Analyze trade-offs between model size, inference time, and accuracy

## Dataset

**CIFAR-10:** A dataset of 60,000 32×32 color images in 10 classes, with 6,000 images per class.
- Training set: 50,000 images
- Test set: 10,000 images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Download: [Kaggle CIFAR-10 Dataset](https://www.kaggle.com/competitions/cifar-10/data)

## Model Architecture

**ResNet (Residual Networks):** Deep neural networks with skip connections that enable training of very deep networks.

Variants tested:
- ResNet-50 or ResNet-18

## Methodology

1. **Baseline Training:** Train ResNet models on CIFAR-10 without pruning
2. **Structured Pruning:** Apply channel-level or filter-level pruning
3. **Fine-tuning:** Re-train pruned models to recover performance
4. **Evaluation:** Compare accuracy, model size, and inference time


## Contact & Support

For questions or issues related to this project, please contact any team member through the HUST email system.

---

**Last Updated:** March 2026
