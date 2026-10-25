# 🖋️ HandwritingLab — AI Handwriting Detector & Toolkit

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **HandwritingLab** is an all-in-one AI handwriting recognition, document OCR, arithmetic solver, and forensic signature verification system. It features **both a modern, responsive web application** (usable from your desktop or phone touch screen) and a **native desktop GUI**, powered by pre-trained neural networks and computer-vision pipelines that run purely on CPU with no GPU required!

---

## 🌟 What Makes This Project Unique & Special?

Unlike simple single-purpose MNIST tutorials or generic OCR scripts, **HandwritingLab** combines **six full-fledged intelligence engines** into a unified, modular toolkit:

1. ✍️ **Multi-Digit Handwriting Recognition**
   - Custom LeNet-style PyTorch CNN (**99.1% accuracy** on MNIST).
   - Intelligently segments connected components so you can draw multi-digit numbers (e.g. `42`, `1984`) and inspect what the computer sees.
2. 🔤 **Alphabet & Character Classifier**
   - Trained on EMNIST Balanced (**87.8% multi-class test accuracy** across 47 classes).
   - Instant real-time character recognition with probability distributions for top predictions.
3. 🧮 **Handwritten Math Expression Solver**
   - Draw math sums like `7 + 3 * 2` directly on the canvas.
   - Segments digits and operators (`+`, `-`, `*`, `/`), refines digits via the 99.1% digit model, parses operator precedence, and safely evaluates results using Python AST (no insecure `eval()`).
4. 📄 **Hybrid Dual-Engine OCR**
   - **Printed Mode:** Ultra-fast printed document text extraction using Tesseract OCR.
   - **Handwriting Mode:** Line-segmented handwriting reading powered by Microsoft's Transformer Vision model (`microsoft/trocr-base-handwritten`).
5. 🧑 **Biometric Writer Identification**
   - Enrol individual writers using handwriting image samples.
   - Extracts comprehensive feature embeddings (HOG + Local Binary Patterns + PCA) to match unknown handwriting samples to enrolled authors via cosine similarity.
6. ✒️ **Forensic Signature Verification & Heatmap Analysis**
   - Calibrated against the authoritative CEDAR dataset (**ROC-AUC 96.4%, EER ~11%**).
   - Compares questioned signatures against genuine samples and generates a **color-coded visual difference heatmap** (identifying stroke deviations and slant changes) alongside an exportable forensic report.

---

## 🚀 Quick Start & How to Run

Pre-trained models are already included in the `models/` directory (~31 MB total), so **everything runs immediately out of the box** without training.

### 1. Clone the Repository
```bash
git clone https://github.com/BHARATH9629/Handwriting_Detector.git
cd Handwriting_Detector
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment (optional but recommended)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment (Optional)
Copy `.env.example` to `.env`:
```bash
copy .env.example .env     # Windows
# cp .env.example .env     # Linux/macOS
```
You can customize the host, port, or specify a custom Tesseract OCR path if needed.

---

### 4. Run the Application

You have two interactive ways to run the project:

#### 🌐 Option A: Modern Web Application (Recommended)
Run the Flask server:
```bash
python web_app.py
```
- Open your browser at **`http://localhost:5000`**
- **Mobile Friendly:** The terminal prints your local network IP (e.g., `http://192.168.1.X:5000`). Connect your smartphone to the same Wi-Fi and draw directly on your touch screen!

#### 💻 Option B: Native Desktop Window (Tkinter)
Launch the standalone desktop GUI:
```bash
python handwritinglab.py
```

---

## 🧪 Running the Tests

To verify the models, preprocessing algorithms, and inference pipelines:
```bash
python run_tests.py
```
This runs smoke tests verifying digit CNN, multi-digit sequence parsing, letter CNN, calculator evaluation, and image preprocessing.

---

## 📊 Measured Model Benchmarks

All models were evaluated on independent, held-out test datasets:

| Engine / Model | Dataset / Benchmark | Test Score | Notes |
|----------------|---------------------|------------|-------|
| **Digit CNN** | MNIST Test Set | **99.1% Accuracy** | 98% on machine-rendered multi-font glyphs |
| **Letter CNN** | EMNIST Balanced (47 classes) | **87.8% Accuracy** | ~97% on rendered glyphs |
| **Calculator CNN** | 14-Class Combined Dataset | **99.2% Accuracy** | Re-verifies digits with dedicated 99.1% model |
| **Signature Verifier** | CEDAR Signature Dataset | **96.4% ROC-AUC** | ~11% Equal-Error-Rate on unseen signers |
| **Printed OCR** | Tesseract Engine v5 | Fast & reliable | Automatically detected in system path |
| **Handwriting OCR** | TrOCR Transformer | Line-by-line SOTA | Runs on CPU, splits lines via ink-density |

---

## 📂 Project Architecture

```
Handwriting_Detector/
│
├── web_app.py               # Flask Web Application (REST APIs + canvas endpoints)
├── handwritinglab.py        # Desktop GUI (Tkinter application with drawing pad)
├── run_tests.py             # Automated test suite
│
├── hlab/                    # Core AI & Computer Vision Package
│   ├── __init__.py          # Package initialization
│   ├── preprocess.py        # Binarization, deskewing, centering & MNIST framing
│   ├── features.py          # HOG, LBP, Hu moments & geometrical descriptors
│   ├── digits.py            # PyTorch Digit CNN architecture & inference
│   ├── letters.py           # EMNIST Letter CNN architecture & inference
│   ├── calc.py              # Math segmentation, classification & safe AST evaluator
│   ├── signature.py         # Signature comparison & writer-independent verifier
│   ├── writer_id.py         # Author identification & profile management
│   ├── ocr.py               # Tesseract OCR integration & graceful fallbacks
│   ├── handwriting_ocr.py   # TrOCR transformer line-splitting & text extraction
│   └── report.py            # Difference heatmaps & exportable reports
│
├── models/                  # Committed Pre-trained Models (~31 MB total)
│   ├── digit_cnn.pt         # Digit CNN weights (PyTorch)
│   ├── letter_cnn.pt        # Letter CNN weights (PyTorch)
│   ├── calc_cnn.pt          # Calculator symbols weights (PyTorch)
│   └── signature_model.joblib # CEDAR-trained verifier bundle
│
├── templates/
│   └── index.html           # Web app UI template
├── static/
│   ├── style.css            # Dark-theme styling & responsive layout
│   └── app.js               # Canvas drawing & API handlers
│
├── train_digits.py          # Script to train Digit CNN on MNIST
├── train_letters.py         # Script to train Letter CNN on EMNIST
├── train_calc.py            # Script to train Calculator model
├── evaluate_signature.py    # Script to train/evaluate Signature Verifier on CEDAR
│
├── requirements.txt         # Python project dependencies
├── .env.example             # Configuration template
└── README.md                # Project documentation
```

---

## ⚙️ Configuration (.env)

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `5000` | Port for the web application |
| `HOST` | `0.0.0.0` | Host interface (`0.0.0.0` enables LAN/phone access) |
| `DEBUG` | `False` | Flask debug mode |
| `SECRET_KEY` | `handwriting-detector-secret-key...` | Flask session secret key |
| `TESSERACT_CMD` | `""` (auto-detected) | Custom path to `tesseract.exe` if not on PATH |

---

## 🔧 Optional Enhancements

- **Printed Text OCR:** Install Tesseract OCR:
  - Windows: `winget install UB-Mannheim.TesseractOCR`
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`
- **Handwriting OCR (TrOCR):** When using the handwriting OCR mode for the first time, Hugging Face will automatically download `microsoft/trocr-base-handwritten` (~1.3 GB) to your user cache directory (`~/.cache/huggingface`).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
