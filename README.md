# Student Placement Prediction – Full Report Card

This project predicts **PlacementStatus** (Placed/Not Placed) from student academic/training features and serves a **Streamlit web app** that shows a **full report card** (probability gauge, radar chart vs cohort medians, quick tips).

## Dataset
- Input file used: `data/raw/placementdata.csv`
- Detected target column: `PlacementStatus`
- Detected features (auto-mapped): ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 'AptitudeTestScore', 'ExtracurricularActivities', 'PlacementTraining']

If any expected columns are missing or named differently, update `config.json` or the code in `src/train.py` accordingly.

## Quickstart (VS Code)

### 1) Open in VS Code & create venv
```bash
python -m venv venv
# If you're in Git Bash:
source venv/Scripts/activate
# If you're in CMD/PowerShell:
# venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train the model
```bash
python src/train.py
```
Artifacts saved to `models/`: `preprocessor.joblib`, `classifier.joblib`, `threshold.json`, `cohort_medians.json`

### 3) Run the web app
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

## Notes
- The training script uses **Stratified K-Fold CV**, threshold tuning for **best F1 (Placed)**, and saves **cohort medians** for the radar chart.
- The app reads the saved artifacts and produces a **probability gauge**, **radar chart**, and **actionable tips**.
- For SHAP local explanations, enable the optional section in the app (commented for speed).
