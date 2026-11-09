# Student Performance Prediction System

Classroom prototype that estimates whether a student is **at risk of failing
the final Mathematics grade**.

## Problem
Academic risk is often noticed only after poor results. This prototype explores
early identification using available student indicators.

## Dataset
UCI Machine Learning Repository - Student Performance (Mathematics file,
`student-mat.csv`): https://archive.ics.uci.edu/dataset/320/student+performance

## Prediction target
`G3 < 10` = AT RISK, `G3 >= 10` = NOT AT RISK. `G3` is the target only and is
never used as an input feature.

## Technologies
Python, pandas, scikit-learn (Random Forest), Streamlit, joblib, GitHub.

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py      # downloads data, trains and saves student_model.pkl
streamlit run app.py       # opens http://localhost:8501
```
Use the "Download sample input CSV" button in the app to test uploads.

`app.py` creates `student_model.pkl` and `student_data.csv` automatically on
first run if they are missing (for example on Streamlit Cloud). If the dataset
download fails, the app lets you upload `student-mat.csv` instead.

## Limitations
- Trained on one Portuguese secondary-school dataset; it may not generalize to
  other institutions or populations.
- G1 and G2 are strong predictors, so the model flags risk late in the term.
- Predictions are not diagnoses and must not be the sole basis for decisions.
