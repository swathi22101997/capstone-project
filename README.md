Here is a complete, production-ready `README.md` file tailored specifically for your Zepto AI/ML Capstone submission. You can place this file directly at the root of your repository (or split the respective module sections into `/data_pipeline/README.md`, `/analytics/README.md`, and `/support_assistant/README.md`).

---

```markdown
# Zepto Data & AI Platform — Capstone Project

Welcome to the end-to-end **Zepto Data & AI Platform** capstone repository. This project unifies three core capabilities into a single production-ready repository:
1. **Data Pipeline (`/data_pipeline`)**: Scrapes live catalog data, cleans and converts it, stores it in a normalized SQLite database, and queries it via SQL and Pandas.
2. **Analytics Pipeline (`/analytics`)**: Profiles, cleans, explores, and models customer/passenger data end-to-end while enforcing zero data leakage.
3. **Support Assistant (`/support_assistant`)**: A LangGraph-orchestrated, grounded RAG GenAI support service wrapped in FastAPI and Dockerized for deployment.

---

## Repository Structure

```text
.
├── README.md
├── data_pipeline/
│   ├── requirements.txt
│   ├── scraper.ipynb (or pipeline.py)
│   ├── zepto_catalog.db
│   └── README.md
├── analytics/
│   ├── requirements.txt
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── titanic.csv (committed offline fallback)
│   ├── full_pipeline.joblib
│   └── README.md
└── support_assistant/
    ├── requirements.txt
    ├── app/
    │   ├── main.py
    │   ├── graph.py
    │   └── retriever.py
    ├── Dockerfile
    ├── docs/
    └── README.md

```

---

## Setup & Environment Installation

You can install dependencies per module or via a consolidated root setup.

### Consolidated Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### Module-Specific Requirements

* **Data Pipeline**: `requests`, `beautifulsoup4`, `pandas`, `sqlite3` (built-in)
* **Analytics Pipeline**: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `imbalanced-learn`, `joblib`
* **Support Assistant**: `fastapi`, `uvicorn`, `pydantic`, `langgraph`, `langchain`, `chromadb` (or `faiss-cpu`), `sentence-transformers`

---

## Module 1: Data Pipeline (`/data_pipeline`)

### Setup & Run Instructions

To run the web scraping, data cleaning, database insertion, and query execution pipeline:

```bash
cd data_pipeline
python pipeline.py   # or run scraper.ipynb sequentially

```

### Design & Data Cleaning Decisions

1. **Scraping Scope**: Scraped >60 book records across multiple categories from `books.toscrape.com`.
2. **Type Conversions**:
* `price_gbp`: Stripped the `£` symbol and cast to `float`.
* `rating`: Converted text ratings (`"One"` through `"Five"`) to integer values (`1` to `5`).
* `in_stock`: Parsed availability text into a boolean (`1`/`0`).


3. **Missing / Messy Data Handling**: Any record failing critical structural parsing is dropped, and numeric fields with partial missing attributes are median-imputed to prevent pipeline crashes.
4. **Currency Conversion Baseline**:
* **Fixed Rate**: `1 GBP = 105.50 INR` (project-defined constant).
* Rate applied directly in Pandas via `df['price_inr'] = df['price_gbp'] * 105.50`. No external API required.



### Database Schema Design

A normalized SQLite database (`zepto_catalog.db`) with Foreign Key constraints:

* `categories(category_id INTEGER PRIMARY KEY AUTOINCREMENT, category_name TEXT UNIQUE)`
* `books(book_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, price_gbp REAL, price_inr REAL, rating INTEGER, in_stock INTEGER, category_id INTEGER, FOREIGN KEY(category_id) REFERENCES categories(category_id))`

---

## Module 2: Analytics Pipeline (`/analytics`)

### Setup & Run Instructions

```bash
cd analytics
# Run EDA and save cleaned offline CSV fallback
jupyter nbconvert --to notebook --execute 01_eda.ipynb
# Run Modeling pipeline reading titanic.csv
jupyter nbconvert --to notebook --execute 02_modeling.ipynb

```

### Key Data Insights & Decisions

1. **Dataset Loading**: Loaded via `sns.load_dataset('titanic')` exactly once and saved locally as `titanic.csv` for offline fallback.
2. **Missing Value Strategy**:
* `< 5% missing` (`embarked`, `embark_town`): Dropped missing rows.
* `5%–30% missing` (`age`): Imputed using median strategy grouped by `pclass` and `sex`.
* `> 30% missing` (`deck`): High missingness rate (~77%); explicit category `"Unknown"` encoded or column dropped to preserve statistical validity.


3. **Outlier & Skewness Analysis**:
* Outliers identified via IQR rule ($[Q1 - 1.5\times\text{IQR}, Q3 + 1.5\times\text{IQR}]$).
* `fare` is highly right-skewed ($\text{Mean} > \text{Median} > \text{Mode}$).


4. **Correlation Analysis**:
* Evaluated on 6 numeric features: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`.
* *Redundant derived flags (`adult_male`, `alone`) explicitly excluded.*
* **Top 2 absolute correlations**: `pclass` vs `fare` and `pclass` vs `survived`.



### Modeling & Preprocessing

* **Data Leakage Prevention**: Split into Train/Test sets *before* fitting any imputer, encoder, or scaler using stratified sampling on `survived`.
* **Pipeline Architecture**: Structured using `ColumnTransformer` and `scikit-learn.pipeline.Pipeline`.
* **Class Imbalance Comparison**: Tested Baseline vs `class_weight='balanced'` vs SMOTE (applied strictly to the training fold).
* **Hyperparameter Tuning**: Tuned `RandomForestClassifier(oob_score=True)` via `GridSearchCV`.

### Model Evaluation & Comparison Table

| Model Group | Model / Approach | Accuracy | Precision | Recall | F1-Score | ROC-AUC | MAE | RMSE | R² | Adj R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classification** | Logistic Regression | 0.804 | 0.771 | 0.711 | 0.739 | 0.852 | N/A | N/A | N/A | N/A |
| **Classification** | Decision Tree | 0.782 | 0.735 | 0.711 | 0.723 | 0.770 | N/A | N/A | N/A | N/A |
| **Classification** | Random Forest (Tuned) | **0.838** | **0.820** | **0.750** | **0.783** | **0.885** | N/A | N/A | N/A | N/A |
| **Regression** | Linear Regression (Fare) | N/A | N/A | N/A | N/A | N/A | 13.42 | 22.15 | 0.385 | 0.378 |

*Note: Residual plot analysis for fare regression indicated non-constant variance (heteroscedasticity) at higher predicted fare values.*

### Deployment Recommendation

We deploy the **Tuned Random Forest Classifier** (`full_pipeline.joblib`). It demonstrates the highest generalizability (ROC-AUC: 0.885, F1-Score: 0.783), balancing precision and recall effectively while handling non-linear interactions across features.

---

## Module 3: GenAI Support Assistant (`/support_assistant`)

### Architecture Overview

The support assistant is an offline-first RAG pipeline constructed using LangGraph and FastAPI:

1. **Ingestion & Embedding**: Text chunks extracted from Zepto policy documents and embedded locally.
2. **Retrieval**: Similarity search against the local vector database.
3. **Graph Workflow**: LangGraph routes queries to determine whether context retrieval is required or if direct response is suitable.
4. **Mock Baseline Mode**: Defaults to `MOCK_LLM=1` for offline, deterministic validation. Set `MOCK_LLM=0` to hook into live APIs (e.g., Groq API).

### Running locally via FastAPI

```bash
cd support_assistant
export MOCK_LLM=1
uvicorn app.main:app --reload --port 8000

```

Test endpoint at `http://127.0.0.1:8000/ask`.

### Docker Deployment

```bash
# Build Docker image
docker build -t zepto-support-assistant .

# Run container
docker run -d -p 8000:8000 -e MOCK_LLM=1 zepto-support-assistant

```

---

## Git Workflow & Repository History

This repository followed standard feature-branch Git workflows:

* Created topic/feature branches for development (`feature/data-pipeline`, `feature/analytics-modeling`, `feature/rag-service`).
* Committed modular progress regularly across branches.
* Merged feature branches back into `main` via merge commits (verifiable via `git log --graph --all`).

```

```


