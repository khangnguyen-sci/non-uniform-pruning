# Non-uniform Structured Pruning for Efficient Diffusion-based Real-world Image Super-resolution. (ICIP 2026)

Official implementation of our ICIP 2026 paper:

## Overview

We study resolution-dependent redundancy in the Stable Diffusion U-Net backbone for real-world image super-resolution (Real-ISR). Instead of applying uniform channel pruning across all layers, we introduce **stage-wise structured pruning aligned with encoder–decoder resolution symmetry**.

Our analysis shows that deeper bottleneck stages can be pruned more aggressively while preserving reconstruction quality.

**Key results**

- 34% parameter reduction (456M → 301M)
- 10.3% U-Net latency reduction
- 6.9% end-to-end inference speed improvement
- Competitive performance on RealSR and DRealSR benchmarks

---

## Method

The Stable Diffusion U-Net is partitioned into resolution stages:

R0 → R1 → R2 → R3 → MID

Instead of uniform pruning:

R0 = R1 = R2 = R3 = RMID

we apply resolution-aware pruning:

R0 = R1 = R2 = 0.75  
R3 = RMID = 0.25

This preserves skip-connection compatibility while reducing computation.

---

## Requirements

Recommended environment:

- Python 3.10  
- PyTorch 2.4+  
- CUDA 11.8+

Install dependencies:

- pillow>=10.0.0
- opencv-python-headless==4.11.0.86
- tqdm==4.65.2
- omegaconf==2.3.0
- torch==2.4.1
- torchvision==0.19.1
- torchaudio==2.4.1
- xformers==0.0.28.post1
- fairscale==0.4.13
- loralib==0.1.2
- timm==0.9.16
- pyiqa==0.1.13
- transformers==4.37.2
- diffusers==0.32.2
- peft==0.13.2
- pytorch-lightning==2.4.0

---

## Datasets

Evaluation datasets:

DIV2K-Val  
RealSR  
DRealSR

Place datasets under:

datasets/RealSR  
datasets/DRealSR  
datasets/DIV2K_Val

---

## Inference

Run:

python test.py \
--LR_dir datasets/RealSR/LR \
--SR_dir results/RealSR

---

## Efficiency Measurement

Latency measured on:

NVIDIA Tesla V100 (32GB)  
FP16 precision  
Input resolution: 128×128  

Timing protocol:

50 warm-up iterations  
1000 averaged runs using CUDA events

We report:
Inference (Time)
Parameters  
MACs  
U-Net latency (Time*)  
End-to-end latency

---


This work is still being updated and will have an official web version for the compressed models.
