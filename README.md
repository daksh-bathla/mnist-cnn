# MNIST Handwritten Digit Recognition with CNN

A complete deep learning project featuring a 3-layer Convolutional Neural Network trained on the MNIST dataset, with an interactive Streamlit web app for predictions.

## Overview

- **Model**: 3-layer CNN (PyTorch)
- **Dataset**: MNIST (60,000 training, 10,000 test images)
- **Accuracy**: **99.33%** on test set
- **Architecture**: Conv2D → MaxPool → Conv2D → MaxPool → Conv2D → Dense → Output

## Project Structure

```
├── mnist_cnn.py          # Training script (generates plots + trains model)
├── streamlit_app.py      # Interactive web app
├── requirements.txt      # Python dependencies
├── model_accuracy_loss.png   # Training curves
├── confusion_matrix.png      # Performance metrics
└── predictions.png          # Sample predictions
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

Run the CNN training script:

```bash
python mnist_cnn.py
```

**Output:**
- Trains for 5 epochs on CPU
- Generates 3 PNG visualizations:
  - `model_accuracy_loss.png` — accuracy & loss curves
  - `confusion_matrix.png` — 10×10 confusion matrix
  - `predictions.png` — 5 random test predictions

## Interactive Web App

Launch the Streamlit app to visualize results and make predictions:

```bash
streamlit run streamlit_app.py
```

**Features:**
- 📊 Overview: model metrics & training curves
- 📈 Model Metrics: confusion matrix & architecture
- 🖊️ Interactive Prediction: upload images or see random test predictions

## Model Architecture

```
SimpleCNN(
  Conv2D(1→32, 3×3) → ReLU → MaxPool(2×2)
  Conv2D(32→64, 3×3) → ReLU → MaxPool(2×2)
  Conv2D(64→128, 3×3) → ReLU → Flatten
  Dense(128) → Dropout(0.5) → Dense(10, softmax)
)
```

## Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 99.33% |
| Test Loss | 0.0252 |
| Training Time | ~2 min (CPU) |
| Parameters | ~140K |

## Technologies

- **PyTorch** — deep learning framework
- **Streamlit** — interactive web app
- **Scikit-learn** — metrics & confusion matrix
- **Matplotlib & Seaborn** — visualization

## Live Demo

Deployed on Streamlit Cloud: [MNIST Digit Recognition](https://mnist-digit-recognition.streamlit.app)

---

Built with ❤️ • [Source Code](https://github.com/dakshbathla/mnist-cnn)
