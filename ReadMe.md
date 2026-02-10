# NSL-Fingerspell: Deep Generative Motion Modeling for Nepali Sign Language

An AI-powered system that generates high-fidelity 3D fingerspelling animations from Nepali text input using Transformer-based sequence generation and anatomical physics constraints.

![Project Status](https://img.shields.io/badge/Status-In--Development-yellow)
![AI Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Computer Vision](https://img.shields.io/badge/CV-MediaPipe-blue)

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Technical Innovation](#technical-innovation)
- [Installation](#installation)
- [Dataset Pipeline](#dataset-pipeline)
- [Usage](#usage)
- [Future Work](#future-work)

---

## Overview

Fingerspelling is a critical component of Nepali Sign Language (NSL), used for names, technical terms, and words without dedicated signs. This project treats fingerspelling as a **Text-to-Motion** generation task. By leveraging a Transformer Encoder-Decoder architecture, the system translates Nepali characters into a continuous stream of 3D coordinates (225-dimensional vectors) representing human pose and hand geometry.

---

## System Architecture

The project follows a modular pipeline design:

1.  **Data Engineering:** Keypoint extraction from raw .MOV videos using MediaPipe.
2.  **Normalization:** Wrist-centric hand scaling and shoulder-centered pose alignment to ensure signer-independence.
3.  **Generative Model:** A Transformer-based architecture with a custom MLP Text Encoder.
4.  **Sequential Inference:** A character-by-character generation loop with autoregressive history and stabilization mixing.

---

## Key Features

- **Sequential Word Generation:** Generates complex Nepali words (e.g., "घर", "नमस्ते") by stitching individual character motions with realistic "holds" and transitions.
- **Anatomical Constraints:** Integrated loss functions that enforce physical rules, preventing finger overlap and "inside-out" hand flipping.
- **High-Speed Processing:** Optimized for 60fps data to capture the rapid transitions inherent in fingerspelling.
- **Custom Annotation Tool:** A built-in OpenCV-based GUI for frame-perfect segmentation of continuous signing videos.

---

## Technical Innovation: Multi-Task Loss Function

To achieve "Perfect Shapes," the model is trained using a composite loss function:

- **Weighted MSE Loss:** Prioritizes finger landmarks (40x weight) over torso landmarks.
- **Cosine Orientation Loss:** Uses normal vectors to ensure the palm faces the camera correctly.
- **Angular & Bone Loss:** Prevents fingers from "collapsing" by maintaining constant bone lengths and realistic joint angles.
- **Static Body Constraint:** Penalizes unnecessary torso movement, ensuring the avatar remains stable during signing.

---

## Installation

### Prerequisites

- Python 3.9+
- NVIDIA GPU (RTX 4070 Laptop GPU used for development)
- CUDA 11.8+

### Setup

```bash
git clone https://github.com/yourusername/NSL-Fingerspell.git
cd Nepali-Sign-Language
pip install -r requirements.txt
```

---

## Dataset Pipeline

The project uses a structured CLI to manage data building across four distinct categories:

| Stage                | Command                                                       |
| :------------------- | :------------------------------------------------------------ |
| **Build Vowels**     | `python main.py --stage build --data vowel --type single`     |
| **Build Consonants** | `python main.py --stage build --data consonant --type single` |
| **Annotate Multi**   | `python src/annotator.py --mode vowel`                        |
| **Build Multi**      | `python main.py --stage build --data vowel --type multi`      |

---

## Usage

### 1. Training the Model

To start the training engine (with built-in early stopping and learning rate scheduling):

```bash
python -m main --stage train
```

### 2. Generating Sign Language

To generate a `.npz` motion file from Nepali text:

```bash
python -m main --stage generate
```

_Input Example:_ `घर`
_Output:_ `experiments/generated_output.npz`

### 3. Visualization

Generated results can be viewed using the centered skeleton visualizer provided in the project notebooks.

---

## Future Work

- **Blender Integration:** Developing a Python wrapper for Blender to map generated `.npz` coordinates to a 3D Mesh using Inverse Kinematics (IK).
- **Facial Expressions:** Integrating MediaPipe Face Mesh to include non-manual markers (mouthings) during signing.
- **Real-time API:** Optimizing the Transformer for real-time inference in mobile applications.

---

### Author

**[Bishesh Giri]**

---
