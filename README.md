**Bias Detection and Debiasing System**

**Project Overview**

This project aims to detect and mitigate bias in textual data using machine learning models, including fine-tuned BERT and GPT-based approaches. The system identifies biased text, highlights bias probability at the token level, and generates debiased alternatives while preserving the original meaning.

Features

Bias Detection: Uses a BERT-based model to classify text as biased or unbiased.

Token-wise Bias Analysis: Provides visual representation of token importance and bias probability.

Bias Validation: Evaluates model performance using standard metrics like precision, recall, and F1-score.

Text Debiasing: Leverages a fine-tuned LLM (T5 model) to generate unbiased text alternatives.

Data Processing: Handles dataset preprocessing and embedding generation for training and evaluation.
**
**Project Structure****

📂 project_root
│── app_BERT_LLM.py                  # Streamlit application for bias analysis
│── bias_model_BERT.py                # BERT-based bias detection model
│── bias_model_BERT_validation.py     # Model evaluation and validation
│── unbias_model_LLM.py               # T5-based debiasing model
│── data.py                            # Data processing and preparation
│── Data_Fetch.ipynb                   # Data fetching and exploration notebook
│── GPT_Bias_Classification_Fine_Tuning.ipynb  # Fine-tuning GPT for bias classification
│── GPT_based_Debaising.ipynb          # Debiasing using GPT models
│── PreGenerate_Embeddings.ipynb       # Pre-generating embeddings for bias analysis
│── SVM_Bases_Bais_Detection.ipynb     # Bias detection using SVM
│── models/                            # Directory for trained models
│── data/                              # Raw datasets
│── processed_data/                     # Processed datasets
│── logs/                               # Log files

Installation

Prerequisites

Ensure you have Python 3.8+ installed along with the necessary dependencies.

Setup Environment

Clone the repository:

git clone <repo-url>
cd <project-root>

Install dependencies:

pip install -r requirements.txt

Usage

Running the Bias Detection App

To launch the Streamlit application, run:

streamlit run app_BERT_LLM.py

Running Bias Validation

To validate the bias detection model:

python bias_model_BERT_validation.py

Data Preprocessing

To preprocess datasets:

python data.py

Model Training

Fine-tuning the BERT model for bias classification and GPT/T5 for debiasing can be done using the respective Jupyter Notebooks provided.

Results and Metrics

The system provides performance metrics like precision, recall, and F1-score, along with visualization tools to analyze bias detection performance.

Contributors

Ankit

License

This project is licensed under the MIT License.

