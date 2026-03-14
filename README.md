# 🍌 Banana Quality Inspector

An internal-style Streamlit dashboard for predicting banana quality grades using a trained Gradient Boosting classifier.

Built for the IE MBDS Machine Learning II group assignment.

---

## Folder Structure

```
banana_app/
│
├── app.py                      ← Streamlit application (main entry point)
├── train_model.py              ← Training pipeline: loads CSV, trains, saves model
├── utils.py                    ← Shared helpers: feature engineering, metadata, validation
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
├── banana_quality_dataset.csv  ← ⚠️  Place your dataset here (see step 2)
│
└── model/                      ← Created automatically by train_model.py
    └── banana_pipeline.joblib  ← Saved sklearn Pipeline (created at training time)
```

---

## Quick Start

### Step 1 — Clone / copy the project files

Make sure all four files (`app.py`, `train_model.py`, `utils.py`, `requirements.txt`) are in the same folder.

### Step 2 — Add the dataset

Copy `banana_quality_dataset.csv` into the project folder (same level as `app.py`).

### Step 3 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Train the model

This only needs to be done **once**. It creates `model/banana_pipeline.joblib`.

```bash
python train_model.py
```

You should see output like:

```
[data]  Loaded 1,000 rows × 16 columns
[cv]   Accuracy : 0.9125 ± 0.0231
[test] Accuracy : 0.9050
[save] Pipeline saved to  model/banana_pipeline.joblib
✓  Training complete. You can now launch the Streamlit app.
```

### Step 6 — Launch the app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## How to use the app

1. Use the **sidebar sliders and dropdowns** to enter the banana batch measurements.
2. Press **Run Quality Prediction** to log a prediction to the history table.
3. The **Quality Prediction** panel shows the predicted grade and class probabilities.
4. The **Operational Recommendation** panel gives routing and pricing guidance.
5. The **Key Quality Drivers** panel compares your inputs against the dataset average.
6. The **Prediction History** table records all predictions made in the session.
7. Use **Export CSV** to download the inspection log.

---

## Model details

| Item | Value |
|------|-------|
| Algorithm | GradientBoostingClassifier (scikit-learn) |
| Hyperparameters | n_estimators=200, max_depth=5, learning_rate=0.1 |
| Classes | Unripe · Processing · Good · Premium |
| CV Accuracy | ~91–93% (5-fold stratified) |
| Test Accuracy | ~90% |
| Imbalance handling | `compute_sample_weight('balanced')` |
| Feature pipeline | Median imputation + OneHotEncoder in ColumnTransformer |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Model file not found` error | Run `python train_model.py` first |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your venv |
| Port already in use | Run `streamlit run app.py --server.port 8502` |
| Plotly charts not showing | Upgrade: `pip install --upgrade plotly streamlit` |
