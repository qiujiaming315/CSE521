# CSE521: Open-World Object Detection for Smart Kitchen

## Overview

This repository implements the course project of CSE 521, AI & IoT for Medicine, Fall 2023.

## Requirements

We recommend a recent Python 3.7+ distribution of [Anaconda](https://www.anaconda.com/products/individual) with `numpy`, `torch`, and `scikit-learn` installed.

See ``INSTALL`` for detailed instructions on installation.

## Open-World Detection

The implementation consists of two parts: synthetic image generator and object detector based on YOLOv5. Detailed guidance on how to run the code is available in ``HOW-TO-RUN``.

#### Synthetic Dataset

We use both a [real-world dataset](https://drive.google.com/drive/folders/1Ab2LuM90zhq58YGDxqmKwxQtMzdiW9RP?usp=drive_link) and a synthetic dataset to train our object detector. You can run our ``main.py`` to create synthetic images together with the annotations associated with the objects.

#### Real-Time Detection with YOLOv5

Our object detector is based on [YOLOv5](https://github.com/ultralytics/yolov5). We provide a patch to the public YOLOv5 implementation under the folder ``yolo_patch`` for our task-specific services, including the open-world object detection algorithm, a user-friendly color palette to visualize detection results, and an AI-assisted voice service.
