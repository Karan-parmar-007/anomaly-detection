# Setup & Execution Guide

## System Requirements

### Required Configuration

- **Python**: 3.12 (Required)
- **RAM**: 8GB minimum
- **Storage**: 5GB free space

### Development Environment

This project was developed on:

- **OS**: WSL Linux Ubuntu
- **CPU**: AMD Ryzen 7
- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **RAM**: 32GB
- **Storage**: 5GB for project

---

## Installation

### Step 1: Verify Python 3.12

```bash
python --version
```

Should show: `Python 3.12.x`

Download from: https://www.python.org/downloads/

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages (pandas, numpy, matplotlib, seaborn, statsmodels, scipy, scikit-learn, tensorflow with CUDA support).

---

## Running the Analysis

### Step 3: Execute Notebook

Place these files together:

```
your_folder/
├── train.csv
└── main.ipynb
```

Launch Jupyter:

```bash
jupyter notebook
```

Open the notebook and click: `Cell` → `Run All`

**Runtime**:

- With GPU (RTX 3060): ~5 minutes
- With CPU: ~10 to 15 minutes
