# Efficient Financial Sentiment Classification Using FinBERT and LoRA
📘 Overview

This repository contains the implementation and report for the project Parameter-Efficient Fine-Tuning of FinBERT for Financial Sentiment Classification.
The goal is to compare Full Fine-Tuning and LoRA (Low-Rank Adaptation) on the Financial PhraseBank dataset for classifying financial news sentences as positive, neutral, or negative.

LoRA achieves near-parity performance with Full Fine-Tuning while updating roughly 1% of parameters only, demonstrating the effectiveness of parameter-efficient learning for financial NLP.

🧠 Features

Financial PhraseBank preprocessing and label encoding
TF-IDF + Logistic Regression baseline
FinBERT fine-tuning using Hugging Face Transformers
LoRA (PEFT) integration for efficient adaptation
Evaluation metrics: Accuracy, Macro-F1, ROC-AUC, PR-AUC
Efficiency and parameter-count comparison
Error analysis with representative misclassifications
