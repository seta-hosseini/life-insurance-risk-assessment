# 📊 Life Insurance Risk Assessment Prototype

This repository contains a two-person team project developed during the **Karestan Javan** experiential learning program at the **University of Tehran** in **Spring 2024**.  
The model is trained on **1,000 synthetic policyholder records** generated to reflect plausible demographic, health, and financial profiles.  

## 📝 Overview

1. **Synthetic Data Generation**  
   Generated policyholder profiles with 16 features (age, BMI, smoking, diabetes, occupational risk, family and personal medical history, activity level, income, debts, and more), with mortality outcomes drawn from a logistic risk model.  

2. **Machine Learning from Scratch**  
   - Implemented **feature standardization** (StandardScaler) without external libraries.  
   - Implemented **logistic regression** trained with **batch gradient descent** (configurable learning rate and epochs).  
   - Evaluated the model on an 80/20 train–test split.  

3. **Single-Profile Assessment**  
   Enter one applicant's details to get a predicted mortality risk and a suggested payout based on coverage amount and policy term.  

4. **Batch Simulation**  
   Simulate any number of random profiles and summarize the distribution of predicted risks and payouts (mean, median, standard deviation, min, max).  

5. **Desktop Interface**  
   Built a full desktop GUI with **CustomTkinter**: dashboard with live training progress, advanced settings for hyperparameters, and an in-app explanation of the calculations.  

---

## 📁 Repository Structure

```
├── src/              # Desktop application (single Python file)
├── docs/             # Methods overview (Persian)
├── requirements.txt
└── LICENSE
```

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python src/life_insurance_app.py
```

The model trains automatically in the background at startup. The interface is in Persian.

## 🛠️ Tools

Python · CustomTkinter · Tkinter · Pillow
