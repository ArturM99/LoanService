# Loan Approval Prediction Service

Сервис предсказывает, будет ли одобрена заявка на кредит (`Loan_Status`), по анкетным данным заявителя: пол, семейное положение, доход, размер и срок кредита, кредитная история и т.д.

## Структура проекта

```
.
├── main.py                    # FastAPI-сервис для инференса
├── Model/
│   ├── pipeline.py            # обучение и автоматический выбор лучшей модели
│   ├── loan_pipe.pkl          # обученная модель (создаётся pipeline.py)
│   └── Data/
│       ├── loan_train.csv     # обучающие данные (не входят в репозиторий)
│       ├── form_LP001014.json # пример входных данных для /predict
│       └── form_LP001024.json # пример входных данных для /predict
└── requirements.txt
```

## Установка

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Минимальный набор зависимостей:

```
fastapi
uvicorn
pandas
scikit-learn
joblib
```

## Данные

Файл `Model/Data/loan_train.csv` в репозиторий не включён. Ожидаемые колонки: `Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, `Loan_Status` (целевая переменная — `Y`/`N`).

## Обучение модели

```bash
python Model/pipeline.py
```

Скрипт:

1. Читает `data/loan_train.csv`, убирает `Loan_ID`, переводит `Loan_Status` в бинарный вид (`Y` → 1, `N` → 0).
2. Числовые признаки заполняет медианой и масштабирует (`StandardScaler`), категориальные — модой и кодирует (`OneHotEncoder`).
3. Перебирает три модели — `LogisticRegression`, `RandomForestClassifier`, `MLPClassifier` (нейросеть с тремя скрытыми слоями) — через 4-фолдовую кросс-валидацию и выбирает лучшую по accuracy.
4. Обучает лучшую модель на всех данных и сохраняет вместе с метаданными в `loan_pipe.pkl` (через `joblib`).

Пример вывода:

```
model: LogisticRegression, acc_mean: 0.XXXX, acc_std: 0.XXXX
model: RandomForestClassifier, acc_mean: 0.XXXX, acc_std: 0.XXXX
model: MLPClassifier, acc_mean: 0.XXXX, acc_std: 0.XXXX
best model: RandomForestClassifier, accuracy: 0.XXXX
```

## Запуск API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger-документация: `http://localhost:8000/docs`

### Эндпоинты

| Метод | Путь        | Описание                                   |
|-------|-------------|----------------------------------------------|
| GET   | `/status`   | Проверка работоспособности сервиса            |
| GET   | `/version1` | Метаданные модели (тип, автор, дата обучения, accuracy) |
| POST  | `/predict`  | Предсказание одобрения кредита                |

### Пример запроса `POST /predict`

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

### Пример ответа

```json
{
  "Loan_ID": "LP001014",
  "Result": 1.0
}
```

`Result` — предсказанный класс: `1.0` (кредит одобрен) или `0.0` (отказ).

## Важно

- Лучшая модель выбирается автоматически по кросс-валидации на этапе обучения — какая именно модель победила, видно в выводе `pipeline.py` или через эндпоинт `/version1`.
- `main.py` и `Model/pipeline.py` используют одну и ту же логику препроцессинга через `ColumnTransformer`, поэтому обработка данных при обучении и инференсе идентична.
- В папке `Model/Data/` лежат готовые примеры входных данных (`form_LP001014.json`, `form_LP001024.json`) — их можно использовать для быстрой проверки эндпоинта `/predict` через Swagger UI или `curl`.
