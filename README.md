# Mini Search Engine

A mini search engine built in Python aimed at illustrating Information Retrieval concepts such as indexing and ranking using Clean Code principles.

## Structure
- `src/core/preprocessing.py`: Text preprocessing (Tokenization, Lemmatization, Stemming, Stop-words removal).
- `src/core/`: Contains core indexing and ranking modules (Incidence Matrix, Inverted Index, Positional Index, Ranker).
- `src/features/`: Contains advanced features (Query Expansion, Spelling Correction, Snippet Generation).
- `data/documents_split/`: The directory intended for `.txt` files to be indexed.
- `frontend/app.py`: Streamlit web application.
- `tests/`: Unit Tests to ensure code correctness.

## Requirements
Make sure to install the required packages by running:
```bash
pip install -r requirements.txt
```

## Usage
### Web Interface
You can run the web interface using Streamlit:
```bash
streamlit run frontend/app.py
```

### CLI
You can also use the command line interface to interact with the engine:
```bash
python main.py "your query" --dir data/documents_split
```
