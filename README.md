# EcoMaintenance

EcoMaintenance is a multimodal industrial maintenance assistant designed for grounded, citation-backed troubleshooting.

It can work with:

- Technical manuals
- Scanned documents
- Tables
- Wiring diagrams
- Engineering drawings
- Equipment images
- Multiple documents at once

## Key Features

- Multi-document technical search
- Citation-backed troubleshooting answers
- Wiring diagram and drawing analysis
- Table and specification reasoning
- OCR support for scanned manuals
- Manufacturer-source fallback
- Evidence viewer with source links
- Mobile-friendly technician interface
- Grounded response generation using retrieved evidence

## Tech Stack

- Python
- FastAPI
- OpenAI API
- Tavily API
- PyMuPDF
- BM25 retrieval
- HTML
- CSS
- JavaScript

## Run Locally

1. Clone the repository

2. Create a virtual environment:

```bash
python -m venv .venv

3. Activate it on Windows:
.venv\Scripts\activate

4. Install dependencies:
pip install -r requirements.txt

5. Create a .env file based on .env.example
6. Add your API keys:
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
OPENAI_REFINEMENT_MODEL=gpt-5.6-luna
OPENAI_DRAWING_MODEL=gpt-5.6-terra

7. Start the application:
python -m uvicorn main:app --host 0.0.0.0 --port 8000

8. Open:
http://127.0.0.1:8000


Built for ABB EarthHacks 2026 — Theme 2: Multimodal Maintenance Intelligence Agent.
