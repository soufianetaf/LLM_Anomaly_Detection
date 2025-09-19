# Log Anomaly Detection using Synthetic Log Generation with LLM

This repository focuses on improving anomaly detection in system logs by generating synthetic logs using Large Language Models (LLMs). The approach combines LSTM-based models and synthetic log generation to enhance the detection of anomalies, especially in datasets with class imbalance.

## Project Overview

The project consists of the following stages:
1. **Data Retrieval**: Fetching datasets from Zenodo and GitHub (HDFS and BGL).
2. **Parsing**: Cleaning and structuring raw logs into an analyzable format.
3. **Log Generation**: Using models like DistilGPT-2 for fine-tuning and synthetic log generation.
4. **Anomaly Detection**: Using LSTM models to classify logs into normal or anomalous categories.
5. **Word Embedding**: Using BERT for word embedding to enhance model performance.

### Image Representation of the Workflow

The following diagram illustrates the workflow of the project:

![Pipeline Workflow](path/to/pipline_gen.gif)

## Getting Started

To run the project, follow these steps:

### Prerequisites

- Python 3.x
- Required libraries:
  - TensorFlow
  - PyTorch
  - Hugging Face Transformers
  - scikit-learn
  - NumPy
  - pandas
  - matplotlib

Install the dependencies using `pip`:

```bash
pip install -r requirements.txt
