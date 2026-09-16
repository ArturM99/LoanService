# Loan Approval Prediction Service

The service predicts whether a loan application will be approved (`Loan_Status`), based on applicant data: gender, marital status, income, loan amount and term, credit history, etc.

## Project structure

```
.
├── main.py                    # FastAPI inference service
├── Model/
│   ├── pipeline.py            # training and automatic best-model selection
│   ├── loan_pipe.pkl          # trained model (created by pipeline.py)
│   └── Data/
│       ├── loan_train.csv     # training data (not included in the repo)
│       ├── form_LP001014.json # example input data for /predict
│       └── form_LP001024.json # example input data for /predict
└── requirements.txt
```

## Installation

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Minimal dependencies:

```
fastapi
uvicorn
pandas
scikit-learn
joblib
```

## Data

The `Model/Data/loan_train.csv` file is not included in the repo. Expected columns: `Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, `Loan_Status` (target — `Y`/`N`).

## Training the model

```bash
python Model/pipeline.py
```

The script:

1. Reads `data/loan_train.csv`, drops `Loan_ID`, converts `Loan_Status` to binary (`Y` → 1, `N` → 0).
2. Fills numerical features with the median and scales them (`StandardScaler`); categorical features are filled with the mode and encoded (`OneHotEncoder`).
3. Compares three models — `LogisticRegression`, `RandomForestClassifier`, `MLPClassifier` (a neural net with three hidden layers) — via 4-fold cross-validation and picks the best one by accuracy.
4. Trains the best model on the full dataset and saves it together with metadata to `loan_pipe.pkl` (via `joblib`).

Example output:

```
model: LogisticRegression, acc_mean: 0.XXXX, acc_std: 0.XXXX
model: RandomForestClassifier, acc_mean: 0.XXXX, acc_std: 0.XXXX
model: MLPClassifier, acc_mean: 0.XXXX, acc_std: 0.XXXX
best model: RandomForestClassifier, accuracy: 0.XXXX
```

## Running the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger docs: `http://localhost:8000/docs`

### Endpoints

| Method | Path        | Description                                              |
|--------|-------------|------------------------------------------------------------|
| GET    | `/status`   | Service health check                                        |
| GET    | `/version1` | Model metadata (type, author, training date, accuracy)      |
| POST   | `/predict`  | Predict loan approval                                        |

### Example `POST /predict` request

```json
{
  "Loan_ID": "LP001014",
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "1",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 1500,
  "LoanAmount": 128,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Semiurban"
}
```

### Example response

```json
{
  "Loan_ID": "LP001014",
  "Result": 1.0
}
```

`Result` — predicted class: `1.0` (loan approved) or `0.0` (rejected).

## Notes

- The best model is automatically selected via cross-validation during training — which model won can be seen in the `pipeline.py` output or via the `/version1` endpoint.
- `main.py` and `Model/pipeline.py` use the same preprocessing logic via `ColumnTransformer`, so data handling is identical during training and inference.
- The `Model/Data/` folder contains ready-made example input files (`form_LP001014.json`, `form_LP001024.json`) that can be used to quickly test the `/predict` endpoint via Swagger UI or `curl`.
-e 
---
🇷🇺 [Читать на русском](https://github.com/ArturM99/LoanService/tree/RU)
