# KDD Flask App

## Project Structure

```text
api/
  index.py                 # Vercel serverless entrypoint
app.py                     # Local Flask entrypoint
kdd_app/
  __init__.py              # Flask app factory
  routes.py                # Web routes
  services/
    ml_service.py          # Prediction model logic
    rag_service.py         # PDF/RAG chat and AI explanations
  templates/
    index2.html            # Main UI
  data/
    models/                # Trained model artifacts
    rag/                   # PDF and FAISS vector store
requirements.txt           # Single dependency file for local and deploy
vercel.json                # Vercel routing/build config
```

Use only `requirements.txt` for installs and deployment. The old split requirement files were duplicates and have been removed.
