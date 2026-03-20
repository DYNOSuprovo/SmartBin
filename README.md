# Smart Waste Detection, Segregation & Bin Fill Estimation System

> **AI‑powered waste classification, automated bin segregation, and approximate fill‑level estimation — fully software‑based, designed for later hardware extension.**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit)

---

## 📋 Project Overview

This project implements a complete end‑to‑end AI system that:

1. **Classifies waste** into 10 categories using deep learning (pretrained CNNs)
2. **Simulates smart segregation** into virtual dustbin compartments
3. **Estimates approximate fill level** from image‑area analysis
4. **Maintains logs and analytics** for every classification event
5. **Provides an interactive dashboard** via Streamlit

### Key Highlights
- 🏆 Multi‑model experiment framework targeting **maximum accuracy (~90%+)**
- 🔬 Two‑phase training: frozen backbone → full fine‑tune
- 📊 Comprehensive evaluation with Grad‑CAM explainability
- 🗑️ Realistic virtual dustbin simulation
- 🌐 Interactive Streamlit web dashboard

---

## 🗂️ Project Structure

```
SDIS/
├── config.py                            # Central configuration
├── 01_data_preparation.py               # Dataset inspection & split
├── 02_model_training_max_accuracy.py     # Multi‑model training framework
├── 03_model_evaluation_and_comparison.py # Post‑training analysis
├── 04_smart_dustbin_simulation.py        # Virtual dustbin simulation
├── streamlit_app.py                      # Interactive web dashboard
├── utils/
│   ├── __init__.py
│   ├── dataset_utils.py                 # Data loading, transforms, visualization
│   ├── training_utils.py                # Model factory, training loop, checkpoints
│   ├── inference_utils.py               # Evaluation, confusion matrix, Grad‑CAM
│   └── simulation_utils.py             # Virtual bin, fill estimation, logging
├── models/                              # Saved model checkpoints
├── logs/                                # Training & simulation logs
├── outputs/                             # Plots, metrics, reports
├── dataset_split/                       # Train/val/test split (auto‑generated)
├── standardized_256/                    # Source images (256×256)
└── README.md
```

---

## 📊 Dataset

**10 waste classes**, ~12,259 images total:

| Class | Count | Bin Mapping | Biodegradable? |
|---|---|---|---|
| battery | 756 | hazardous_bin | ❌ |
| biological | 699 | organic_bin | ✅ |
| cardboard | 1,411 | paper_bin | ✅ |
| clothes | 1,892 | textile_bin | ❌ |
| glass | 1,736 | glass_bin | ❌ |
| metal | 930 | metal_bin | ❌ |
| paper | 1,336 | paper_bin | ✅ |
| plastic | 1,597 | plastic_bin | ❌ |
| shoes | 1,449 | textile_bin | ❌ |
| trash | 453 | trash_bin | ❌ |

**Split:** 70% train / 15% val / 15% test (stratified per class).

---

## 🏋️ Training Strategy

### Models Compared
| Model | Params | Notes |
|---|---|---|
| ResNet18 | 11.2M | Lightweight baseline |
| ResNet50 | 23.5M | Deeper, stronger features |
| EfficientNet‑B0 | 5.3M | Efficient architecture |
| MobileNetV2 | 3.4M | Mobile‑optimized |
| DenseNet121 | 7.0M | Dense connections |
| ConvNeXt‑Tiny | 28.6M | Modern CNN, strong accuracy |

### Two-Phase Training
1. **Phase 1 — Head Only** (5 epochs): freeze backbone, train classifier head at 10× LR
2. **Phase 2 — Full Fine‑Tune** (25+ epochs): unfreeze all layers, cosine LR scheduler

### Optimization
- **Optimizer:** AdamW with weight decay
- **Scheduler:** Cosine annealing
- **Loss:** CrossEntropy with label smoothing (0.1) + class weights
- **Early stopping** with patience=7
- **Augmentations:** RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter

---

## 📈 Evaluation

- Test‑set accuracy, precision, recall, F1
- Normalized confusion matrix
- Per‑class metrics table
- Misclassification gallery
- Grad‑CAM explainability visualizations
- Multi‑model comparison charts

---

## 🗑️ Smart Dustbin Simulation

### Virtual Bin Compartments
| Bin | Waste Types |
|---|---|
| plastic_bin | plastic |
| paper_bin | cardboard, paper |
| metal_bin | metal |
| glass_bin | glass |
| organic_bin | biological |
| textile_bin | clothes, shoes |
| hazardous_bin | battery |
| trash_bin | trash |

### Fill Estimation (Approximate)
Since there is no physical hardware, fill is estimated from image analysis:
1. Isolate foreground using Otsu thresholding + morphology (OpenCV)
2. Compute foreground‑to‑image area ratio
3. Map to size category → fill contribution:

| Size | Area Ratio | Fill Contribution |
|---|---|---|
| very_small | < 5% | +2% |
| small | 5–15% | +5% |
| medium | 15–35% | +10% |
| large | > 35% | +18% |

> ⚠️ **Note:** This is an *approximate visual estimation*, not true volume measurement.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install torch torchvision timm tqdm scikit-learn matplotlib seaborn
pip install pandas numpy Pillow opencv-python
pip install streamlit plotly   # for dashboard
```

### 2. Prepare dataset
```bash
python 01_data_preparation.py
```

### 3. Train models
```bash
python 02_model_training_max_accuracy.py
```

### 4. Evaluate
```bash
python 03_model_evaluation_and_comparison.py
```

### 5. Run simulation
```bash
python 04_smart_dustbin_simulation.py
```

### 6. Launch dashboard
```bash
streamlit run streamlit_app.py
```

---

## 🔮 Future Scope — Hardware Extension

This system is designed so that adding real hardware requires **minimal code changes**:

| Component | Software Version | Hardware Version |
|---|---|---|
| Camera Input | Image upload / webcam | Pi Camera / USB cam |
| Classification | Same PyTorch model | Same model on Raspberry Pi / Jetson |
| Bin Assignment | Virtual bins | Servo motors open correct lid |
| Fill Estimation | Image area heuristic | Ultrasonic sensor / ToF sensor |
| Alerts | Console / UI | Buzzer / LED / IoT notification |
| Logging | JSON / CSV files | Cloud database (Firebase / AWS IoT) |

**Hardware migration path:**
1. Replace `predict_waste()` input from file → camera frame
2. Replace `VirtualBin.add_item()` → send GPIO signal to servo
3. Replace `estimate_fill_contribution()` → read ultrasonic sensor
4. Deploy model on edge device (ONNX / TensorRT)

---

## ⚠️ Limitations

- **Software‑only prototype** — no physical dustbin hardware
- **Fill estimation is approximate** — based on image area, not true volume
- **Dataset-dependent accuracy** — performance may vary on different waste images
- **No real‑time continuous monitoring** — processes one image at a time
- **Single‑object assumption** — designed for one waste item per image

---

## 👨‍💻 Tech Stack

- **PyTorch** — deep learning framework
- **timm** — pretrained model zoo
- **torchvision** — transforms & ImageFolder
- **scikit‑learn** — metrics & evaluation
- **OpenCV** — image processing for fill estimation
- **Matplotlib / Seaborn** — static visualisations
- **Streamlit / Plotly** — interactive dashboard
- **Pandas / NumPy** — data manipulation

---

## 📄 License

This project is for educational and portfolio purposes.
