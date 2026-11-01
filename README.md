# Baymax — AI-Based Expert System for Medical Diagnosis

Baymax is an academic prototype of an AI-based medical diagnosis expert
system. It combines rule-based reasoning, probabilistic inference,
heuristic search, and machine learning into a single Flask web
application with a Baymax-inspired user interface.

> Educational prototype. Not a substitute for professional medical advice,
> diagnosis, or treatment.

---

## Features

- **Knowledge base** of diseases and symptoms (`knowledge_base/`).
- **Rule-based inference engine** with weighted symptom matching.
- **Probabilistic reasoning** via Bayes' Theorem (Naive Bayes).
- **Best-First Search** over candidate diseases with pluggable
  heuristics (Bayesian, rule-based, hybrid).
- **Machine learning models**: Decision Tree, Random Forest, and
  Multinomial Naive Bayes, with saved/loadable model bundles.
- **Synthetic dataset generator** producing a balanced, reproducible
  symptom-to-disease dataset aligned with the Bayesian engine.
- **Evaluation module** that computes accuracy, precision, recall,
  F1-score, classification reports, confusion matrix plots, and
  model comparison tables.
- **Flask web interface** with:
  - a Baymax-inspired CSS-only character,
  - a dropdown-driven symptom selector with removable chips,
  - a transparent diagnosis result page that shows reasoning from
    every component.
- **JSON API** for programmatic access (`/api/symptoms`,
  `/api/diagnose`, `/healthz`).

---

## Project structure
Baymax/
├── app.py
├── requirements.txt
├── README.md
│
├── knowledge_base/
│ ├── diseases.json
│ └── symptom_list.json
│
├── data/
│ └── medical_dataset.csv (generated)
│
├── models/
│ ├── decision_tree_bundle.pkl (generated)
│ ├── random_forest_bundle.pkl (generated)
│ └── naive_bayes_bundle.pkl (generated)
│
├── results/
│ ├── metrics.txt (generated)
│ ├── classification_report.txt (generated)
│ ├── confusion_matrix.png (generated)
│ ├── model_comparison.csv (generated)
│ └── experiment_notes.txt (generated)
│
├── scripts/
│ └── generate_dataset.py
│
├── src/
│ ├── init.py
│ ├── knowledge_base.py
│ ├── utils.py
│ ├── rule_engine.py
│ ├── bayes_engine.py
│ ├── search_engine.py
│ ├── ml_model.py
│ ├── evaluation.py
│ └── diagnosis_pipeline.py
│
├── templates/
│ ├── index.html
│ └── result.html
│
└── static/
├── css/style.css
├── js/script.js
└── images/

---

## Requirements

- Python 3.11 (recommended)
- pip

Python packages are listed in `requirements.txt` and include Flask,
pandas, numpy, scikit-learn, joblib, matplotlib, and seaborn.

---

## Setup

From the project root:

```bash
# 1. Create and activate a virtual environment
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

Generate the synthetic dataset

python scripts/generate_dataset.py

This creates data/medical_dataset.csv (default: 600 rows, 8 diseases,
18 symptoms, fixed seed for reproducibility).

Useful options:
python scripts/generate_dataset.py --cases-per-disease 100
python scripts/generate_dataset.py --seed 7 --output data/medical_dataset.csv

Train the machine learning models
python -c "from src.ml_model import train_all_models; r = train_all_models(); [print(name, 'acc=', round(m['accuracy'], 3), 'f1=', round(m['f1_weighted'], 3)) for name, m in r.items()]"

This trains and saves:

models/decision_tree_bundle.pkl
models/random_forest_bundle.pkl
models/naive_bayes_bundle.pkl

Run the full evaluation
python -c "from src.evaluation import run_full_evaluation; out = run_full_evaluation(primary_model='random_forest'); [print(label, '->', path) for label, path in out['files'].items()]"
This populates the results/ directory with metrics, classification
reports, a confusion matrix image, and a model comparison table.

Run the website
python app.py
Then open:
http://127.0.0.1:5000/
You can:

Pick symptoms from the dropdown and click + Add.
Remove individual chips or clear all.
Click Diagnose to see the top diagnosis, alternative
candidates, the reasoning from each AI component, and the final
ensemble weights.
To stop the server, press Ctrl + C.

JSON API
GET  /api/symptoms — returns the canonical symptom list.
POST /api/diagnose — runs the diagnosis pipeline.
Request body:
{ "symptoms": ["fever", "cough", "fatigue", "headache"] }
GET  /healthz — lightweight health check.

Configuration (environment variables)
BAYMAX_DEBUG=1 — enable Flask debug mode.
BAYMAX_HOST — host to bind (default 127.0.0.1).
BAYMAX_PORT — port to bind (default 5000).
BAYMAX_ML_MODEL — default ML model
(decision_tree, random_forest, or naive_bayes).

Disclaimer
Baymax is an academic prototype intended for educational
demonstration. It must not be used as a substitute for professional
medical advice, diagnosis, or treatment. If your symptoms are severe
or persistent, please contact a qualified healthcare professional.

To setup in another pc:
Install python 3.11 - https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
Important: On the first installer screen, check the box:

[x] Add python.exe to PATH
Then click Install Now.

Verify:

py -3.11 --version
You should see something like:

Python 3.11.x

Step 2: Unzip the project
Unzip Baymax.zip somewhere convenient, for example the desktop. After unzipping, you should have a folder structure like:

C:\Users\<name>\Desktop\Baymax\
    app.py
    README.md
    requirements.txt
    src\
    templates\
    static\
    ...
Step 3: Open a terminal in the Baymax folder
You have two easy ways to do this.

Option A: Use VS Code
Open VS Code.
File → Open Folder → select the unzipped Baymax folder.
Open the integrated terminal (Ctrl +    `).
Option B: Use Command Prompt
Open Command Prompt.
Run:

cd "C:\Users\<name>\Desktop\Baymax"
(Adjust the path to where you actually unzipped the folder.)

Step 4: Create a fresh virtual environment
Inside the project folder:

py -3.11 -m venv venv
This creates a new venv folder, isolated to this PC.

Step 5: Activate the virtual environment
venv\Scripts\activate
Your prompt should change to:

(venv) C:\Users\<name>\Desktop\Baymax>

Step 6: Upgrade pip
python -m pip install --upgrade pip
Step 7: Install dependencies
pip install -r requirements.txt
This installs Flask, pandas, numpy, scikit-learn, joblib, matplotlib, and seaborn into the new venv.

If everything succeeds, you are ready to run the project.

Step 8: (Only if needed) Generate the dataset
If you included data/medical_dataset.csv in the zip, you can skip this step.

If not, run:

python scripts/generate_dataset.py
You should see:

Baymax synthetic dataset generated successfully.
  File:            ...\Baymax\data\medical_dataset.csv
  Rows:            600
  ...

Step 9: (Only if needed) Train the ML models
If you included the models/*.pkl files in the zip, you can skip this step.

If not, run:

python -c "from src.ml_model import train_all_models; r = train_all_models(); [print(name, 'acc=', round(m['accuracy'], 3), 'f1=', round(m['f1_weighted'], 3)) for name, m in r.items()]"
You should see something like:

decision_tree acc= 0.9xx f1= 0.9xx
random_forest acc= 0.9xx f1= 0.9xx
naive_bayes   acc= 0.9xx f1= 0.9xx

Step 10: (Optional) Run full evaluation
If you also want fresh metric files in results/:

python -c "from src.evaluation import run_full_evaluation; out = run_full_evaluation(primary_model='random_forest'); [print(label, '->', path) for label, path in out['files'].items()]"
Step 11: Run the Baymax website

python app.py
You should see:

 * Running on http://127.0.0.1:5000
Open that URL in a browser:

http://127.0.0.1:5000/
Pick some symptoms, click Diagnose, and verify everything works as expected.

To stop the server, press Ctrl + C.

4. Summary checklist
Use this as a quick reference for the other PC:

1. Install Python 3.11 (with "Add to PATH").
2. Unzip Baymax.zip.
3. Open terminal in the Baymax folder.
4. py -3.11 -m venv venv
5. venv\Scripts\activate
6. python -m pip install --upgrade pip
7. pip install -r requirements.txt
8. (Optional) python scripts/generate_dataset.py
9. (Optional) Train models via the one-liner above.
10. (Optional) Run full evaluation.
11. python app.py
12. Open http://127.0.0.1:5000/