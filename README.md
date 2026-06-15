title: ge'ez Text Detection & OCR Pipeline
emoji: 🇪🇹
colorFrom: amber
colorTo: stone
sdk: gradio
app_file: app.py
pinned: false


# 🇪🇹 Amharic Text Detection & Document OCR Pipeline

An end-to-end Document Intelligence application engineered to process low-resource Ge'ez script variants (Amharic). This repository couples classical machine vision image processing pipelines with sequence-to-sequence Deep Learning Vision Transformers.

## 📊 Deep Learning Production Architecture
1. **Spatial Filtering Matrix:** Image inputs are downscaled, converted to single-channel 8-bit matrices, and optimized through a localized Adaptive Gaussian threshold filter to normalize varying document backgrounds.
2. **Sequence Transduction:** Tensors are parsed via a fine-tuned Vision-Encoder-Decoder **TrOCR (Transformer OCR)** layer, generating localized token matrices without necessitating segmented bounding contours.

## 🚀 Live Production Space
Interact directly with the live deployed model interface here: `[PASTE_YOUR_HF_SPACE_URL_HERE]`
