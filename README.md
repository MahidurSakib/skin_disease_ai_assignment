# Skin Disease Detection & LLM Advisor System

A simple, assignment-focused implementation of the **Skin Disease Detection & LLM Advisor System** using:

- **Backend:** FastAPI
- **Frontend:** Streamlit
- **AI/ML:** PyTorch + transfer learning with **ResNet18**
- **LLM:** OpenAI API for prompt-based recommendations
- **Deployment:** Docker file included
- **Database:** Not used, because it is optional in the assignment

## Why this stack was chosen

This project intentionally uses the simplest valid option from the assignment:

- **FastAPI** for the required backend API.
- **Streamlit** for a very fast demo UI.
- **PyTorch + ResNet18** for simple transfer learning without making training too heavy.
- **OpenAI** for short prompt-based recommendations.
- **No database** because the assignment marks it as optional.

## Assignment requirements covered

### Functional requirements
- Image upload API: `POST /analyze_skin`
- Disease classification with confidence score
- LLM-based recommendations
- Real-time API response
- Modular pipeline:
  - image preprocessing
  - CNN-based classification
  - LLM reasoning module

### Technical requirements
- Kaggle skin disease dataset
- Transfer learning model
- Accuracy evaluation
- Confusion matrix generation
- FastAPI endpoint
- Prompt-based LLM recommendations

### Deliverables included in this repo
- GitHub-ready project structure
- README
- API docs file
- Demo UI
- Optional Docker file

## Project structure

```text
skin_disease_ai_assignment/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── config.py
│   ├── llm_service.py
│   ├── model_utils.py
│   ├── preprocessing.py
│   ├── training_utils.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
│
├── prepare_data.py
├── train.py
├── evaluate.py
├── streamlit_app.py
├── API.md
├── Dockerfile
├── requirements.txt
├── .env.example
└── .gitignore
```

## 1) Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows
```bash
.venv\Scripts\activate
```

### macOS / Linux
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2) How to connect your downloaded Kaggle dataset

The assignment dataset is:

```text
https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset
```

After downloading the zip file, extract it.

### Recommended simple placement
Put the extracted dataset inside:

```text
data/raw/
```

If the extracted folder contains `IMG_CLASSES`, your structure can look like this:

```text
skin_disease_ai_assignment/
└── data/
    └── raw/
        └── IMG_CLASSES/
            ├── 1. Eczema 1677/
            ├── 2. Melanoma 15.75k/
            ├── 3. Atopic Dermatitis - 1.25k/
            └── ...
```

### Important
You do **not** need to rename the folders manually.

This project already handles:
- reading the raw class folders
- splitting them into train/val/test
- cleaning class labels for API output

## 3) Prepare train/validation/test split

Run:

```bash
python prepare_data.py --raw-dir data/raw --output-dir data/processed
```

This creates:

```text
data/processed/
├── train/
├── val/
└── test/
```

## 4) Train the model

Run:

```bash
python train.py --data-dir data/processed --epochs 5 --batch-size 32
```

Optional faster training using only the final layer:

```bash
python train.py --data-dir data/processed --epochs 10 --batch-size 16
```

After training, these files are created inside `artifacts/`:

- `best_model.pth`
- `class_names.json`
- `train_history.json`

## 5) Evaluate the model

Run:

```bash
python evaluate.py --data-dir data/processed
```

This creates:

- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`

## 6) Configure the LLM

Copy the env file:

```bash
cp .env.example .env
```

Then set your OpenAI key inside `.env`:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.4-mini
```

### Note
If no OpenAI key is provided, the app still runs with a simple fallback recommendation template.

For the actual assignment demo, using a real OpenAI key is better because the assignment asks for **LLM recommendations**.

## 7) Run the FastAPI backend

```bash
uvicorn app.main:app --reload
```

Backend URLs:

- API base: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## 8) Run the Streamlit demo UI

Open a second terminal and run:

```bash
streamlit run streamlit_app.py
```

By default, the UI sends requests to:

```text
http://127.0.0.1:8000/analyze_skin
```






