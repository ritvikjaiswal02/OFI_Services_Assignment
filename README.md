# Predictive Delivery Optimizer - NexGen Logistics

## Overview
AI-powered logistics tool predicting delivery delays before they occur, built with Python and Streamlit.

## How to Run
1. Open terminal inside project folder
2. Create and activate virtual environment:
   python -m venv venv
   .\\venv\\Scripts\\Activate.ps1
3. Install dependencies:
   pip install -r requirements.txt
4. Train the model (optional):
   python train_model.py data/
5. Run Streamlit app:
   streamlit run app.py

## Key Metrics
- ROC AUC: 0.95
- Accuracy: 93%
- Precision@10: 1.00
- Precision@20: 1.00
- Avg predicted prob (delayed): 0.80
- Avg predicted prob (on-time): 0.10

## Author
Ritvik Jaiswal
OFI AI Internship 2025
