# College Knowledge Assistant (RAG, local model)

A beginner-friendly Retrieval-Augmented Generation (RAG) demo. It searches college policy documents, passes relevant passages to a local language model, and displays source filenames and page numbers. It is designed to abstain when retrieval finds no sufficiently similar passages and to instruct the model to surface conflicting sources instead of choosing one silently.

> **Demo data warning:** The six supplied policies are fictional sample documents, not real college rules.

## Features

- Accepts PDF, TXT, and Markdown documents.
- Requires at least five supported documents before ingestion.
- Extracts PDF text with page numbers.
- Splits text into overlapping chunks.
- Embeds chunks locally with Sentence Transformers.
- Stores normalized vectors in FAISS.
- Uses Ollama and a local Llama model for answer generation; no paid API key is needed.
- Displays supporting source names, page numbers (when available), similarity scores, and excerpts.
- Includes automated unit tests.

## Architecture

1. **Ingestion (`ingest.py`)**: Reads `documents/`, extracts text, splits it into chunks, computes embeddings, and saves `vectorstore/index.faiss` plus `vectorstore/chunks.json`.
2. **Retrieval (`rag_pipeline.py`)**: Embeds the question and searches FAISS for the most similar chunks.
3. **Grounded generation**: Sends the retrieved passages to Ollama with instructions to answer only from evidence, cite source labels, and disclose contradictions.
4. **Abstention**: If no passage passes the similarity threshold, returns a fixed “insufficient information” response without asking the LLM to guess.
5. **UI (`app.py`)**: Streamlit chat interface with expandable source evidence.
6. **Tests (`tests/test_rag.py`)**: Fast unit tests use a fake embedder, so most tests do not download models or require Ollama.

### Limitations

- Similarity thresholds are heuristics, not calibrated confidence values. Tune them on representative questions.
- A language model can still misread evidence or miss a conflict. Validate outputs against the test set and consider adding a contradiction detector for production use.
- The demo uses prompt-based conflict handling rather than a guaranteed logical contradiction detector.
- Scanned/image-only PDFs need OCR before ingestion.
- The supplied text files have no page numbers; the app displays `N/A (text file)`.
- This project is a teaching prototype, not an official college policy authority.

## Prerequisites

- Python 3.11 or 3.12 recommended.
- Internet access for the first installation/model download.
- Ollama installed locally: https://ollama.com/
- Enough disk space for Python packages, the embedding model, and the local LLM.

## Setup (Windows / VS Code)

Open a terminal in the `college-rag-assistant` folder.

### 1. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.12 is not installed, install it from https://www.python.org/downloads/ and select “Add Python to PATH” during setup.

### 2. Install and start the local LLM

Install Ollama from https://ollama.com/. In a terminal run:

```powershell
ollama pull llama3.2:3b
ollama run llama3.2:3b
```

The first command downloads the model. The second starts an interactive session; you can exit it after confirming it works. Ollama normally runs its local API in the background. If the app cannot connect, start Ollama from the Start menu and try again.

To use a different Ollama model, update `OLLAMA_MODEL` in `config.py` and pull that model first. Model size and speed depend on your computer.

### 3. Build the searchable knowledge base

The `documents/` folder already contains six sample text policies. Add at least five `.pdf`, `.txt`, or `.md` files to that folder, then run:

```powershell
python ingest.py
```

The first ingestion downloads the Sentence Transformers embedding model. The resulting local files are written to `vectorstore/`.

Re-run `python ingest.py` whenever documents change.

### 4. Start the UI

```powershell
streamlit run app.py
```

Streamlit prints a local address (usually `http://localhost:8501`). Open it in your browser and ask a question.

## Test questions and expected results

The following are based on the fictional sample documents. Source references may include more than one passage.

| # | Question | Expected result |
|---:|---|---|
| 1 | What is the minimum attendance requirement? | 75% in `attendance_policy.txt` |
| 2 | What time does the library close on weekdays? | 6:00 PM in `library_rules.txt` |
| 3 | What is the exam revaluation application window? | Within 7 working days of results in `exam_policy.txt` |
| 4 | What is the annual scholarship income limit? | ₹3,00,000 in `scholarship_policy.txt` |
| 5 | What is the late registration fee? | ₹500 in `academic_rules.txt` |
| 6 | If attendance is below the required percentage, what could happen and who handles medical exceptions? | Combine `attendance_policy.txt` and `academic_rules.txt`; mention the shortage process and the academic office's documented role |
| 7 | In different words, how much class attendance do students need? | Retrieve the 75% rule despite paraphrasing |
| 8 | Who won the FIFA World Cup in 1998? | Abstain: information is absent from the knowledge base |
| 9 | What is the deadline for the annual scholarship form? | Surface conflict between `scholarship_policy.txt` and `conflicting_policy.txt`; do not silently choose a date |
| 10 | What documents are needed for a scholarship application? | Income certificate and current student ID, if supported by retrieved scholarship text |
| 11 | Can I submit revaluation after 10 working days? | Explain the 7-working-day policy; do not invent an exception |
| 12 | What are the library borrowing limits? | Up to 3 books for 14 days, from `library_rules.txt` |

The exact wording of generated answers can vary. Evaluate whether the factual claim, source, abstention, and conflict behavior match the expected result.

## Automated tests

Run:

```powershell
pytest -q
```

The unit tests cover chunking, text reading, the minimum-document requirement, retrieval matching, low-similarity filtering, missing-answer abstention, source metadata, and missing index handling. They use a fake embedder to run quickly without downloading the embedding model. They do not replace end-to-end evaluation with Ollama.

## Demo plan (3–5 minutes)

1. Show the six files in `documents/`.
2. Run `python ingest.py` and explain chunking, embeddings, and FAISS.
3. Start `streamlit run app.py`.
4. Ask a direct question: “What is the minimum attendance requirement?” Open the source panel.
5. Ask a paraphrased question: “How much class attendance do I need?”
6. Ask a cross-document question about attendance shortage and medical exceptions.
7. Ask an unrelated question about the FIFA World Cup and show the abstention.
8. Ask about the scholarship deadline and show both conflicting sources.
9. Add a new unseen `.txt` document, re-run ingestion, and ask a question whose answer appears only in that file.

## Example technical problem for the presentation

**Problem:** The app returned a plausible answer even though the documents did not contain the required information.

**Cause:** A language model can answer from its pretrained knowledge when retrieval gives weak or irrelevant passages.

**Fix in this project:** Retrieval results are filtered by a configurable similarity threshold. If no passage meets the threshold, the program returns a fixed abstention message without calling the language model. When passages are present, the prompt requires source-grounded answers and explicit conflict reporting.

**Trade-off:** A threshold that is too high can reject valid questions; a threshold that is too low can allow irrelevant passages. Evaluate both answerable and unanswerable questions and tune `MIN_SIMILARITY` in `config.py`. This is a heuristic, not a guarantee against hallucination.

## File guide

- `app.py`: Streamlit UI.
- `config.py`: Model names, file paths, chunk size, overlap, top-k, threshold.
- `ingest.py`: Text extraction, chunking, embedding, and FAISS index creation.
- `rag_pipeline.py`: Retrieval, local generation, source metadata, and abstention.
- `documents/`: Six fictional test policies.
- `tests/test_rag.py`: Automated unit tests.
- `requirements.txt`: Python dependencies.
