# Solar Panel Fault Detection

A deep learning system that classifies solar panel images into six fault categories using a stacking ensemble of three CNN models served via a Flask web application.

---

## Demo

Upload one or more solar panel images through the web interface and get instant fault classification with severity warnings and actionable recommendations.

---

## Fault Classes

| Class | Severity | Action |
|---|---|---|
| Bird-drop | Medium | Cleaning required |
| Clean | OK | No action needed |
| Dusty | Low | Cleaning required |
| Electrical-damage | Critical | Repair required |
| Physical-Damage | Critical | Repair required |
| Snow-Covered | Medium | Cleaning required |

---

## Architecture

```
Solar Panel Images
       |
  Augmentation (Flip, Rotate, ColorJitter, Normalize)
       |
  _____|___________________________________________
  |               |                               |
VGG16        EfficientNetB3              MobileNetV2
(PyTorch)    (PyTorch · WINNER)          (Keras / TF)
30 epochs    20 epochs · Focal loss      12 epochs · Adam
Acc: 87.1%   Acc: 89.3%                 Acc: 87.1%
  |               |                               |
  |_______________|_______________________________|
                  |
         Feature Stack (18 probs)
                  |
       Logistic Regression Meta-Learner
              Acc: 94.6%
                  |
           Flask Web App
    (EfficientNetB3 as primary predictor,
     VGG16 + MobileNetV2 shown as comparative)
```

---

## Results

| Model | Accuracy | Macro F1 | ROC-AUC | ms/image |
|---|---|---|---|---|
| VGG16 | 87.1% | 0.873 | 0.978 | 196 |
| EfficientNetB3 | 89.3% | 0.890 | 0.994 | 42 |
| MobileNetV2 | 87.1% | 0.871 | 0.985 | 23 |
| **Ensemble (LR)** | **94.6%** | **0.942** | **0.998** | — |

---

## Project Structure

```
solar-panel-fault-detection/
├── app.py                        # Flask backend
├── templates/
│   └── index.html                # Web UI
├── static/
│   └── style.css                 # Stylesheet
├── models/
│   ├── vgg16.h5                  # VGG16 checkpoint
│   ├── efficientnet_b3.h5        # EfficientNetB3 checkpoint
│   ├── mobilenet.h5              # MobileNetV2 checkpoint
│   └── meta_learner.pkl          # Stacking ensemble (LR + scaler)
├── dataset_split/
│   ├── train/
│   ├── val/
│   └── test/
├── compare_models.py             # Comparative evaluation of all 3 models
├── stacking_ensemble.py          # Meta-learner training
├── per_class_accuracy.py         # Per-class accuracy analysis
├── confusion_matrix_lr.py        # Confusion matrix for ensemble
└── compare_outputs/
    ├── metrics_summary.csv
    ├── per_class_metrics.csv
    ├── per_class_accuracy.csv
    └── ensemble_results.csv
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/solar-panel-fault-detection.git
cd solar-panel-fault-detection
```

### 2. Install dependencies

```bash
pip install flask torch torchvision tensorflow scikit-learn pillow numpy pandas seaborn matplotlib
```

### 3. Add trained model files

Place the following files inside the `models/` folder:

```
models/vgg16.h5
models/efficientnet_b3.h5
models/mobilenet.h5
models/meta_learner.pkl
```

> To generate `meta_learner.pkl`, run `stacking_ensemble.py` in a Jupyter notebook after training all three base models, then save it:
> ```python
> import pickle
> pickle.dump({'clf': clf, 'scaler': scaler}, open('models/meta_learner.pkl', 'wb'))
> ```

### 4. Run the app

```bash
python app.py
```

Open your browser at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Training

Each model was trained independently. Notebooks/scripts for training are not included in this repo but the key configurations are:

**VGG16**
- Framework: PyTorch
- Epochs: 30
- Optimizer: AdamW with Cosine LR decay
- Loss: CrossEntropy with class weights

**EfficientNetB3**
- Framework: PyTorch
- Epochs: 20
- Optimizer: Adam with ReduceLROnPlateau
- Loss: Focal loss (gamma=2, label smoothing=0.1)
- Sampler: WeightedRandomSampler for class imbalance

**MobileNetV2**
- Framework: Keras / TensorFlow
- Epochs: 12
- Optimizer: Adam (lr=1e-4)
- Base layers frozen

---

## Evaluation

Run the comparison script after training all three models:

```bash
python compare_models.py
```

This prints a full comparative report to console and saves CSVs to `compare_outputs/`.

To train the stacking ensemble and generate `meta_learner.pkl`, run `stacking_ensemble.py` in a Jupyter notebook.

---

## Web App Features

- Drag-and-drop upload for single or multiple images
- Per-image result card with:
  - Predicted fault class and confidence percentage
  - Severity badge (ok / medium / critical)
  - Action warning (No action needed / Cleaning required / Repair required)
  - All 6 class probability bars
  - It gives a recommendation like 'clean' or 'maintances'
- Summary bar showing total images, fault count, critical count, and average confidence

---

## Requirements

```
Python >= 3.9
torch >= 2.0
torchvision >= 0.15
tensorflow >= 2.11
scikit-learn >= 1.2
flask >= 2.3
pillow >= 9.0
numpy
pandas
seaborn
matplotlib
```

---

## Notes

- On Windows, avoid re-importing `torch` or `tensorflow` in separate Jupyter cells after initial load — this causes a DLL initialization error. All analysis cells reuse the objects already in memory.
- TensorFlow GPU is not supported on native Windows for TF >= 2.11. Use WSL2 or CPU inference.
- The app falls back to soft-vote averaging if `meta_learner.pkl` is not found.

---

## License

MIT License
