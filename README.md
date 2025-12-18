# Multilingual LLM Bias & Fairness Audit

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Fairness](https://img.shields.io/badge/Focus-AI%20Ethics-green)

## Project Overview
As Large Language Models (LLMs) are deployed globally, **Algorithmic Fairness** becomes a critical safety requirement. This project audits three industry-standard Transformer models (**BERT, DistilBERT, XLM-RoBERTa**) to quantify and mitigate bias in multilingual sentiment analysis.

Using the **Amazon Multilingual Reviews Corpus (MARC)**, we evaluated how different architectures handle fairness across **English, German, and Spanish**. We attempted to mitigate detected biases using **Language-Specific Fine-Tuning** and **Majority-Voting Ensembles**.

###  Key Objectives
1.  **Quantify Unfairness:** Measure Statistical Parity, Equal Opportunity, and Treatment Equality across languages.
2.  **Compare Architectures:** Analyze the trade-offs between efficiency (DistilBERT) and fairness (XLM-RoBERTa).
3.  **Evaluate Mitigation Strategies:** Test if ensembling models reduces or amplifies existing biases.

---

## Project Architecture
This repository is structured as a reproducible Data Science pipeline:

| File | Description |
| :--- | :--- |
| `01_Data_Exploration.ipynb` | Initial analysis of the MARC dataset distribution and baselines. |
| `02a_Train_Multilingual_DistilBERT.ipynb` | Fine-tuning pipeline for **DistilBERT** (Knowledge Distillation). |
| `02b_Train_Multilingual_BERT.ipynb` | Fine-tuning pipeline for the baseline **mBERT** model. |
| `02c_Train_XLMRoberta_and_German.ipynb` | Fine-tuning pipeline for **XLM-RoBERTa** and German Ensemble. |
| `03a_Ensemble_Inference_English.ipynb` | Aggregating model predictions for English reviews. |
| `03b_Ensemble_Inference_Spanish.ipynb` | Aggregating model predictions for Spanish reviews. |
| `04_Global_Cross_Lingual_Fairness.ipynb` | Comparative analysis of all models and metrics. |

---

##  Key Findings & Insights

### 1. XLM-RoBERTa is the "Fairest" Model
Our results confirm that model architecture and pre-training data size significantly impact downstream fairness.
*   **XLM-RoBERTa** (trained on 2.5TB of CommonCrawl) outperformed both BERT and DistilBERT in accuracy and fairness.
*   It achieved near-perfect **Statistical Parity (~50%)** across all languages, whereas BERT struggled with non-English outliers.

### 2. DistilBERT's "Pessimism Bias"
We identified a specific bias in the distilled model:
*   **The Issue:** DistilBERT consistently over-predicted the **Negative class**, especially in English (Treatment Equality disparity of 173%).
*   **Fine-Tuning Effect:** While fine-tuning improved fairness for Spanish and German, it paradoxically **worsened** fairness for English, highlighting the risk of blind fine-tuning.

### 3. The "Ensemble Paradox"
We attempted to mitigate bias by Ensembling (combining predictions from all three models).
*   **Hypothesis:** Errors would cancel out, leading to a fairer system.
*   **Result:** The ensemble **amplified** the biases (especially in Spanish) rather than reducing them.
*   **Why?** The underlying models (BERT-based) shared similar architectural biases. When combined, their correlated errors reinforced the unfairness rather than correcting it.

---

##  Work Contribution & Team

This project was a collaborative effort at the University of Bologna.

**Valeria Izvoreanu (Me)**
*   **Lead Model Engineer:** Fine-tuned and trained **XLM-RoBERTa-base**, which achieved the highest fairness and performance scores in the study.
*   **Pipeline Optimization:** Designed strategies to optimize training efficiency and reduce computational resource usage for the large XLM-R model.
*   **Fairness Analysis:** Led the definition and analysis of equity metrics (Statistical Parity, Equal Opportunity) to evaluate model bias.
*   **Ensemble:** Developed the ensemble strategy for **German** language data.
*   **Ensemble** Lead the development of the global ensemble.

**Paul-Ioan Clotan**
*   Designed the dataset preprocessing pipeline.
*   Fine-tuned and trained **DistilBERT**.
*   Developed the Spanish language ensemble.

**Antonio Gravina**
*   Conducted initial data analysis.
*   Fine-tuned and trained **BERT**.
*   Developed the English language ensemble.
