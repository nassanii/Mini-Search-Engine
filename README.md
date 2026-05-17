# 🔍 Mini Search Engine

A modular, clean-code **Mini Search Engine** built in Python to illustrate fundamental and advanced concepts in **Information Retrieval (IR)**.

---

###  Dataset Information

> [!NOTE]
> **Where is the dataset?**
> The complete datasets are included directly within this project in the [data](./data) folder:
> - **[data/documents/](./data/documents/)**: Contains 22 raw, large text files representing various newsgroup categories (e.g., computer graphics, autos, crypt, space, medicine, religion, etc.).
> - **[data/documents_split/](./data/documents_split/)**: Contains **1,000 smaller pre-split documents** (`doc_1.txt` to `doc_1000.txt`). This split dataset is the default corpus for the **Web Interface (Streamlit)** and is perfect for exploring core index structures and ranking behavior.

---

##  Architecture & Core Components

This project is organized cleanly into core Information Retrieval structures and advanced features:

### 1. Core Modules (`src/core/`)
- **[Preprocessing (`preprocessing.py`)](./src/core/preprocessing.py)**: A robust preprocessing pipeline that converts raw text into clean, normalized tokens using:
  - Tokenization and lowercasing (Case folding)
  - Punctuation removal
  - Stop-word removal (e.g., "and", "the", "is")
  - Word Normalization (supports both **Lemmatization** via WordNet and **Stemming** via Porter Stemmer)
- **[Incidence Matrix (`incidence_matrix.py`)](./src/core/incidence_matrix.py)**: Implements Boolean Retrieval using a binary term-document representation.
- **[Inverted Index (`inverted_index.py`)](./src/core/inverted_index.py)**: Implements standard Dictionary-Posting List mappings, supporting Boolean `AND` query operations.
- **[Positional Index (`positional_index.py`)](./src/core/positional_index.py)**: Tracks exact word positions in each document, enabling **Exact Phrase Searches** (e.g., matching the exact sequence of "computer graphics").
- **[Ranker (`ranking.py`)](./src/core/ranking.py)**: Computes Term Frequency-Inverse Document Frequency (**TF-IDF**) weights for all tokens and ranks matching documents using **Cosine Similarity**.

### 2. Advanced Features (`src/features/`)
- **[Spelling Correction (`spell_checker.py`)](./src/features/spell_checker.py)**: Analyzes query terms for spelling errors using `pyspellchecker`, dynamically customized with the actual vocabulary of the loaded corpus.
- **[Query Expansion (`query_expansion.py`)](./src/features/query_expansion.py)**: Automatically expands query terms with their synonyms using NLTK's **WordNet** corpus.
- **[Snippet Generator (`snippet_generator.py`)](./src/features/snippet_generator.py)**: Dynamically extracts and highlights the most relevant text windows (snippets) containing the query terms for search results.

---

##  Installation & Setup

Follow these simple steps to set up and run the project on your machine:

### 1. Prerequisites
Ensure you have **Python 3.10 or higher** installed on your system.

### 2. Clone/Open the Workspace
Open your terminal in the root directory of the project:
```bash
cd mini_search_engine
```

### 3. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies:

- **macOS/Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **Windows**:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```

### 4. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

> [!TIP]
> **Automatic NLTK Downloads:**
> You do **not** need to manually download NLTK data libraries! The application's preprocessing module will check for and automatically download any missing NLTK resources (`punkt`, `punkt_tab`, `stopwords`, `wordnet`) on its first run.

---

##  How to Run the Project

The Mini Search Engine can be run in two different modes: a rich, interactive Web UI or a lightweight Command Line Interface (CLI).

###  Option A: The Streamlit Web Interface (Recommended)
This is a gorgeous, fully interactive web application where you can enter search queries, toggle query expansion, choose preprocessing methods (stemming vs. lemmatization), and view real-time data structures.

To start the server, run:
```bash
streamlit run frontend/app.py
```

- **Open in Browser**: The terminal will print a local URL (usually `http://localhost:8501`). Open it to access the search engine.
- **Explore Structure**: Click on tabs like **Ranked Retrieval**, **Exact Phrase**, **Inverted Index**, **Incidence Matrix**, and **Positional Index** to visualize exactly how your search query is mapped inside the engine's internal data structures!

---

###  Option B: The CLI (Command Line Interface)
You can query the search engine directly from the command line:

```bash
python main.py "computer graphics" --dir data/documents_split
```

#### CLI Parameters:
* `"query"`: (Required) The terms you want to search.
* `--dir`: (Optional) The directory of files to index. Defaults to `data/documents`. Pass `data/documents_split` to index the split corpus.

---

##  Running Unit Tests

A comprehensive suite of unit tests is included to verify the index and preprocessor logic. To run the tests, execute:

```bash
python -m unittest discover tests
```

---

##  Project Directory Structure

```text
├── data/
│   ├── documents/             # 22 raw newsgroup text documents
│   └── documents_split/       # 1,000 split text documents (default for Web UI)
├── frontend/
│   └── app.py                 # Streamlit Web Application
├── src/
│   ├── core/
│   │   ├── incidence_matrix.py# Incidence Matrix logic
│   │   ├── inverted_index.py  # Inverted Index and Boolean search
│   │   ├── positional_index.py# Positional Index and Phrase search
│   │   ├── preprocessing.py   # Tokenization, Lemmatization, Stemming
│   │   └── ranking.py         # TF-IDF and Cosine Similarity Ranker
│   └── features/
│       ├── query_expansion.py # WordNet Synonyms Expansion
│       ├── snippet_generator.py# Relevant text snippet extraction
│       └── spell_checker.py   # pyspellchecker customized vocabulary correction
├── tests/                     # Unit tests for core engine components
├── main.py                    # Command-line entry point
├── requirements.txt           # Project Python dependencies
└── README.md                  # Project documentation (this file)
```
