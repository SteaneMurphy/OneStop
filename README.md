# OneStop

An AI-powered shopping assistant built as a university project. Users describe what they're looking for in natural language; the assistant finds relevant products using semantic search and responds conversationally using a locally-run language model.

## How it works

The system uses a RAG (Retrieval-Augmented Generation) architecture:

1. The user sends a message via the chat widget
2. The query is vectorised using a sentence-transformer model (MiniLM)
3. Cosine similarity is computed against pre-vectorised product embeddings to find the top 5 most relevant products
4. Those products are injected as context into a prompt
5. A locally-running LLM (SmolLM3-3B) generates a conversational response
6. The response and matched product list are returned to the frontend, which reorders the product grid to surface the matches

## Stack

**Backend**
- Python / FastAPI
- `sentence-transformers` (all-MiniLM-L6-v2) — semantic product search
- `transformers` + `bitsandbytes` — SmolLM3-3B loaded locally in INT4 quantization
- `scikit-learn` — cosine similarity
- NumPy — pre-computed embedding storage

**Frontend**
- React / TypeScript / Vite
- Chat history persisted in `localStorage`

## Setup

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Download the language model
```bash
python setup/download_model.py
```
Downloads SmolLM3-3B (~6GB) from HuggingFace to `./models/SmolLM3-3B`.

### 3. Generate product embeddings
```bash
python setup/embeddings_setup.py
```
Vectorises the product catalogue and saves embeddings to `data/product_vectors.npy`.

### 4. (Optional) Regenerate product images
```bash
python setup/generate_product_images.py
```
Uses Stable Diffusion (sd-turbo) to generate product images from descriptions. Requires a CUDA-capable GPU. Pre-generated images are already included in the repo.

### 5. Start the backend
```bash
uvicorn assistant_api:app --reload
```

### 6. Start the frontend
```bash
cd frontend/OneStop
npm install
npm run dev
```

## Notes
- The LLM runs locally — no API keys or external services required
- INT4 quantization is used to fit the model on consumer hardware
- Category filtering in the sidebar is UI-only and not yet wired to the product grid
