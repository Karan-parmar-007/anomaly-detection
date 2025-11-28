# Time-Series Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive time-series anomaly detection system for retail sales data spanning 9 years (2010-2018) across 7 stores and 10 products. This project implements 15+ detection methods ranging from statistical approaches to deep learning, identifying 171 store-product level anomalies and 23 temporal pattern anomalies using LSTM autoencoder.

## Key Features

- **Multi-Method Approach**: Combines statistical (Z-score, IQR, MAD, STL), machine learning (Isolation Forest, LOF, SVM, DBSCAN), and deep learning (LSTM Autoencoder) methods
- **Hierarchical Analysis**: Detects anomalies at product, store, and store-product combination levels
- **Ensemble Detection**: Uses voting mechanism requiring 3+ methods to agree for high-confidence anomalies
- **Temporal Pattern Recognition**: LSTM autoencoder identifies subtle deviations from learned patterns
- **Comprehensive Visualization**: Interactive plots and anomaly marking techniques

## Dataset

- **Timespan**: 2010-01-01 to 2018-12-31 (3,287 days)
- **Entities**: 7 stores × 10 products = 70 time series
- **Total Records**: 230,090 daily sales observations
- **Features**: Date, Store ID, Product ID, Number Sold

## Installation

### Prerequisites
- Python 3.12
- 8GB RAM minimum
- 5GB free storage

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Karan-parmar-007/anomaly-detection.git
   cd time-series-anomaly-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This installs all required packages including TensorFlow with CUDA support for GPU acceleration.

## Usage

### Running the Analysis

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**:
   - Navigate to `main.ipynb`
   - Click `Cell` → `Run All`

### Runtime Expectations
- **With GPU (RTX 3060 or similar)**: ~5 minutes
- **With CPU**: ~10-15 minutes

## Project Structure

```
├── FireAI_Test_Karan_Parmar_22_11_25.ipynb  # Main analysis notebook
├── train.csv                                # Dataset (230,090 records)
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
├── README_METHODOLOGY.md                    # Detailed technical methodology
└── README_SETUP.md                          # Setup and execution guide
```

## Methodology Overview

### Detection Methods
1. **Statistical Methods**: Z-score, IQR, MAD, STL decomposition
2. **Machine Learning**: Isolation Forest, Local Outlier Factor, One-Class SVM, DBSCAN
3. **Deep Learning**: LSTM Autoencoder for temporal pattern anomalies
4. **Ensemble Approach**: Voting mechanism for high-confidence detection

### Hierarchical Analysis
- **Product-Level**: 20 anomalies detected
- **Store-Level**: 4 anomalies detected  
- **Store-Product Combinations**: 171 anomalies detected
- **Temporal Patterns**: 23 anomalies via LSTM

### Key Findings
- Store 0 has 73 anomalies (43% of total) but was invisible in aggregate analysis
- Product 1 shows persistent issues across multiple years
- October 2018 cluster: Major temporal pattern disruption detected only by deep learning
- Ensemble precision: Requires 3+ methods agreement, resulting in 13 high-confidence anomalies

## Results Summary

| Method Category | Anomalies Detected | Detection Rate |
|-----------------|-------------------|----------------|
| Overall Ensemble | 13 | 0.40% |
| Product-Level | 20 | Varies |
| Store-Level | 4 | Varies |
| Store-Product | 171 | Varies |
| LSTM Autoencoder | 23 | 0.70% |

**Total Unique Anomalies**: 220+ across all methods

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset provided for FireAI assignment
- Built with TensorFlow, scikit-learn, and other open-source libraries
- Developed on WSL Linux Ubuntu with NVIDIA RTX 3060 GPU

## Contact

Karan Parmar - karan.ai.engineer@gmail.com

Project Link: [https://github.com/your-username/time-series-anomaly-detection](https://github.com/your-username/time-series-anomaly-detection)

---

For detailed technical analysis, see [README_METHODOLOGY.md](README_METHODOLOGY.md).  
For setup and execution details, see [README_SETUP.md](README_SETUP.md).