
# Hybrid Search RAG with Citation Verification

A Python-based Retrieval-Augmented Generation (RAG) retrieval pipeline that combines dense vector search and BM25 keyword search to retrieve relevant documents, reranks the results using a cross-encoder, and verifies whether selected evidence supports an answer.

## Project Overview

This project demonstrates a hybrid information retrieval pipeline designed to improve evidence selection and check citation support.

## Architecture

1. Document Loading
2. Document Chunking
3. Dense Vector Retrieval
4. BM25 Keyword Retrieval
5. Reciprocal Rank Fusion (RRF)
6. Cross-Encoder Reranking
7. Best Evidence Selection
8. Citation Verification
9. Final Result: SUPPORTED / NOT SUPPORTED

## Technologies Used

- Python
- Sentence Transformers
- BM25
- NumPy
- Cross-Encoder Reranking

## Project Structure

```text
hybrid-rag/
├── data/
│   └── documents/
│       └── python.txt
├── main.py
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd hybrid-rag
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install sentence-transformers rank-bm25 numpy
```

## Run the Project

```bash
python main.py
```

The first execution downloads the required pretrained models.

## Evaluation

The project includes three sample questions to evaluate evidence retrieval and citation verification.

The final output identifies the best evidence and classifies citation support as:

- SUPPORTED
- NOT SUPPORTED

## Key Features

- Dense semantic retrieval
- BM25 keyword-based retrieval
- Hybrid ranking using Reciprocal Rank Fusion
- Cross-encoder reranking
- Evidence-based citation verification
- Multi-question evaluation

## Author

Yassar Sharief M
