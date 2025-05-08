# Emotion Classification of Speech Audio

This tutorial demonstrates an end-to-end workflow for **automated emotion classification** from speech audio, using the RAVDESS and CREMA datasets. Tailored for social scientists, it covers data organization, preprocessing, feature extraction, model training in PyTorch, and expected inferences—highlighting methodological choices and their justifications, especially around detecting **anger**.

---

## Table of Contents

1. [Overview](#overview)
2. [Datasets](#datasets)
3. [Preprocessing Pipeline](#preprocessing-pipeline)

   * A. Data Cleaning & Organization
   * B. Data Augmentation
   * C. Feature Extraction
   * D. Scaling & Normalization
   * E. Label Encoding & Split
4. [Feature Summary & Dataset Sizes](#feature-summary--dataset-sizes)
5. [Model Architecture](#model-architecture)
6. [Training Loop (PyTorch)](#training-loop-pytorch)
7. [Expected Outcomes & Inferences](#expected-outcomes--inferences)
8. [Justification for Anger Detection](#justification-for-anger-detection)
9. [References](#references)

---

## Overview

We build a system that:

1. **Loads** two benchmark datasets of emotional speech (RAVDESS, CREMA).
2. **Preprocesses** audio via cleaning, augmentation, and fixed-length feature extraction (MFCC, Chroma, Mel-Spectrogram).
3. **Scales** and **encodes** features and labels for neural modeling.
4. **Trains** a simple PyTorch neural network to predict emotion categories.
5. **Evaluates** generalization on held-out test data.

The tutorial emphasizes **why** each step matters for robust, interpretable emotion research in social science contexts.

---

## Datasets

* **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech & Song)**
  • Original speech samples: actor\_01 → actor\_24, 8 emotions × 2 intensities × 2 modalities → \~2,400 WAV files.
* **CREMA-D (Crowd-sourced Emotional Multimodal Actors dataset)**
  • \~7,400 WAV files labeled with 6 basic emotions by 91 actors.

**Total original samples:** \~9,800 audio clips.
**After noise augmentation (doubling):** \~19,600 samples.

---

## Preprocessing Pipeline

### A. Data Cleaning & Organization

1. **File listing:** Recursively scan dataset folders (`glob`, `os.listdir`).
2. **Label parsing:** Extract emotion labels from filenames, map to a unified set (e.g., `disgust`, `fear`, `happy`, `sad`, `anger`, `neutral`).
3. **Error handling:** Skip unreadable or silent files to avoid corrupted inputs.

*Why?* Ensures a clean, consistent corpus free of I/O errors—critical before feature extraction.

### B. Data Augmentation

* **Additive Gaussian noise:**
  $y_{aug} = y + \alpha \cdot \max(|y|) \cdot \mathcal{N}(0,1)$
* **(Optional)** Pitch shifting & time stretching to simulate speaker variability.

*Why?* Increases robustness against real-world recording conditions, and balances class representation when samples are scarce.

### C. Feature Extraction

For each clip (original + noisy):

1. **MFCCs (40 coefficients):** timbral representation aligned with human auditory perception.
2. **Chroma (12 bins):** energy distribution across pitch classes.
3. **Mel-Spectrogram (128 bands):** detailed time-frequency map.

**Concatenate** → **180-dimensional** feature vector.

*Why?* Combines complementary acoustic cues to capture prosody, intonation, and spectral patterns associated with emotion.

### D. Scaling & Normalization

* **StandardScaler:** mean=0, σ=1 per feature dimension.

*Why?* Prevents features with large variances (e.g., spectrogram power) from dominating smaller-scale MFCCs.

### E. Label Encoding & Split

1. **One-Hot Encoding:** converts each emotion label to a binary vector for multi-class cross-entropy.
2. **Stratified 80/20 split:** preserves emotion proportions in train & test sets.

*Why?* Maintains nominal nature of emotion categories and ensures fair evaluation on unseen data.

---

## Feature Summary & Dataset Sizes

| Stage                    | Count    |
| ------------------------ | -------- |
| Original RAVDESS clips   | \~2,400  |
| Original CREMA-D clips   | \~7,400  |
| **Combined originals**   | \~9,800  |
| After noise augmentation | \~19,600 |
| **Train set (80%)**      | \~15,680 |
| **Test set (20%)**       | \~3,920  |
| Feature vector dimension | 180      |

---

## Model Architecture

We implement a PyTorch neural network with:

* **Input layer:** 180 neurons (one per feature).
* **Hidden layers:** 128 → 64 units with ReLU activations.
* **Output layer:** 6 neurons (one per emotion category).
* **Loss:** `CrossEntropyLoss` (maps logits → class probabilities).
* **Optimizer:** `Adam` (lr=0.001).

```python
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)
```

---

## Training Loop (PyTorch)

* **Epochs:** 30
* **Batch size:** 32
* **Procedure:** For each batch, zero gradients, forward pass, compute loss, backpropagate, optimizer step. Track **average loss** per epoch.

```python
for epoch in range(30):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.argmax(dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
```

---

## Expected Outcomes & Inferences

* **Model performance:** >70% test accuracy (varies by emotion).
* **Error analysis:** Identify which emotions (e.g., `fear`, `disgust`) are often confused—insightful for understanding acoustic markers of complex affect.
* **Feature importance:** Analyze learned weights or apply SHAP values to see which MFCCs or spectrogram bands correlate with specific emotions.

From a social science perspective, this enables:

* Large-scale, objective analysis of emotional tone in speech corpora.
* Investigations into demographic or contextual moderators of emotional expression.
* Integration with qualitative studies to triangulate findings.

---

## Justification for Anger Detection

**Anger** is a fundamental emotion with profound social implications:

* **Conflict dynamics:** anger expression predicts escalation in interpersonal and group conflicts.
* **Mental health:** persistent anger can signal distress or aggression risk.
* **Political discourse:** vocal anger shapes persuasion, mobilization, and polarization.

Automatically detecting anger in large speech datasets empowers social scientists to quantify patterns of hostility, evaluate intervention outcomes, and study cultural norms around emotional expression.

---

## References

* Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (**RAVDESS**).
* Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.
* Davis, S., & Mermelstein, P. (1980). Comparison of Parametric Representations for Monosyllabic Word Recognition.
* Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.
* Ellis, D. P. W. (2007). Chroma Feature Tutorial.
