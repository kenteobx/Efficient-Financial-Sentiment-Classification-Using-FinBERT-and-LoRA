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

🗂️ Repository Structure
├── code.ipynb      # Main Colab notebook (code + results)
├── data/                       # Dataset files (Financial PhraseBank subsets)
│   └── Sentences_AllAgree.txt
└── README.md                   # This file

⚙️ Setup & Installation
# Clone repository
git clone https://github.com/kenteobx/Efficient-Financial-Sentiment-Classification-Using-FinBERT-and-LoRA.git
cd finbert-lora-sentiment

# Create environment (optional)
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

If running in Google Colab, simply upload the notebook and dataset files into /content/ or mount Google Drive.

🚀 Running the Project

1. Open code.ipynb in Jupyter Notebook or Google Colab.
   
2. Run all cells sequentially:
    - Data loading and preprocessing
    - Baseline → Full Fine-Tuning → LoRA training
    - Evaluation and efficiency comparison
  
3. View metrics, plots, and misclassified examples at the end of the notebook.

📊 Results Summary
| Model                 | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | % Trainable Params |
| --------------------- | -------- | -------- | ------- | ------ | ------------------ |
| TF-IDF + LogReg       | 0.872    | 0.820    | 0.959   | 0.908  | N/A                |
| FinBERT (Full FT)     | 0.978    | 0.972    | 0.999   | 0.996  | 100 %              |
| FinBERT (LoRA / PEFT) | 0.974    | 0.969    | 0.996   | 0.985  | ≈ 1 %              |

LoRA retains almost identical predictive performance while using 98.8 % fewer trainable parameters.

💡 Key Takeaways
    - Performance: Transformer models vastly outperform classical baselines.
    - Efficiency: LoRA delivers near-parity performance with a fraction of parameters. 
    - Error Behaviour: Models struggle most on forward-looking or comparative statements.
    - Interpretability: Lexical inspection aligns model attention with expected financial tone.

⚖️ Ethical Considerations & Limitations

Financial sentiment models can influence decision-making.
Biases in data or annotation quality may lead to misclassification of market-sensitive information.
Future work should test robustness on multilingual or social-media data and integrate explainability tools such as SHAP or attention visualisation.

🧑‍💻 Acknowledgements
Dataset: Financial PhraseBank
 by M. Takala et al.
Model: ProsusAI FinBERT
PEFT/LoRA: Hu et al. (2022), LoRA: Low-Rank Adaptation of Large Language Models.

