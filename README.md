# Multimodal Vector Search Engine 🔍

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![React](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)
![Tailwind](https://img.shields.io/badge/Tailwind_CSS-38B2AC?logo=tailwind-css&logoColor=white)

An academic and portfolio flagship project demonstrating state-of-the-art **Multimodal Information Retrieval** using **OpenAI's CLIP**, **Facebook AI Similarity Search (FAISS)**, and **FastAPI**. 

This system indexes the 31,000+ image **Flickr30k dataset** and enables millisecond-latency cross-modal searches (Text-to-Image, Image-to-Image, and Image-to-Text).

---

## 🚀 Key Features

- **Dual-Index FAISS Architecture:** Maintains separate highly-optimized inner-product (`IndexFlatIP`) vector spaces for both 31k images and 158k text captions.
- **Cross-Modal Retrieval:** 
  - `Text → Image`: Type a semantic query ("dogs playing in snow") to find visually matching images.
  - `Image → Image`: Upload a photo to find visually and semantically similar images.
  - `Image → Text`: Upload an image to retrieve the most statistically probable ground-truth captions.
- **High-Performance Batching:** Utilizes PyTorch tensor batching for 10x faster dataset ingestion and L2-normalized embedding generation.
- **Academic Evaluation Pipeline:** Built-in validation script (`evaluate.py`) to calculate standard IR metrics: **Recall@1, Recall@5, Recall@10, and Mean Reciprocal Rank (MRR)**.
- **Developer-Focused UI:** A minimalist, dark-themed React + Tailwind v4 interface featuring real-time inference telemetry (Cosine Similarity, Latency, FAISS ID, and Embedding tensor previews).

---

## 🧠 System Architecture

The search engine maps both text and images into the same 512-dimensional continuous vector space. 

1. **Embedding Generation:** Inputs are passed through the frozen `openai/clip-vit-base-patch32` Vision Transformer.
2. **L2 Normalization:** Vectors are L2-normalized to project them onto a unit hypersphere.
3. **Vector Indexing:** FAISS computes the Inner Product (Dot Product). Because the vectors are L2-normalized, this mathematically equates to **Cosine Similarity**, ensuring highly accurate semantic clustering.

---

## 📂 Dataset (Flickr30k)

This project uses the [Flickr30k dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) containing ~31,783 images, each paired with 5 human-annotated ground-truth captions.
- **Automated setup:** The `download_flickr.py` script uses the Kaggle API to automatically download, extract, and clean the dataset.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+
- Node.js 18+
- [Kaggle Account](https://www.kaggle.com/) (for dataset downloading)

### 2. Clone the Repository
```bash
git clone https://github.com/bhaveshjenna/P42-Vector-Search.git
cd P42-Vector-Search
```

### 3. Backend Setup
Open a terminal and set up the Python backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Data Ingestion Pipeline
Download the dataset and generate the FAISS vectors. *(Note: Full ingestion takes some time depending on your CPU/GPU).*

```bash
# 1. Download Flickr30k from Kaggle (Requires Kaggle API token in ~/.kaggle/kaggle.json)
python download_flickr.py

# 2. Dry run (Test the pipeline on just 100 images)
python backend/scripts/ingest_data.py --limit 100

# 3. Full Ingestion (Generates images.index, captions.index, and mapping.json)
python backend/scripts/ingest_data.py
```

### 5. Frontend Setup
Open a **second terminal** and set up the React frontend:
```bash
cd frontend
npm install
npm run dev
```

---

## 🏃‍♂️ Running the Application

To run the application locally, you need both the backend and frontend servers running simultaneously.

**Terminal 1 (Backend):**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*The interactive API documentation (Swagger UI) will be available at: http://localhost:8000/docs*

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```
*The UI will be available at: http://localhost:5173*

---

## 📊 Evaluation & Metrics

This project is built for rigorous academic evaluation. To test the mathematical accuracy of the retrieval engine, run the built-in evaluation script against your generated indexes:

```bash
python backend/scripts/evaluate.py
```

**What it calculates:**
- **Recall@K (R@1, R@5, R@10):** Measures the percentage of times the correct match appears in the top K results.
- **Mean Reciprocal Rank (MRR):** Evaluates the ranking quality (higher is better).

---

## 🏗️ Project Structure

```text
P42-Vector-Search/
├── backend/
│   ├── core/
│   │   ├── config.py         # Environment variables & paths
│   │   └── schemas.py        # Pydantic models for API validation
│   ├── scripts/
│   │   ├── ingest_data.py    # Pipeline to vectorize the dataset
│   │   └── evaluate.py       # Validation and Metrics (Recall/MRR)
│   ├── services/
│   │   ├── clip_service.py   # Wrapper for HuggingFace CLIP model
│   │   └── faiss_service.py  # Wrapper for FAISS vector operations
│   └── main.py               # FastAPI application entry point
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main React interface
│   │   ├── index.css         # Tailwind v4 configuration
│   │   └── main.jsx          # React DOM entry
│   └── vite.config.js        # Vite config with API proxying
├── data/                     # Ignored by git (Images & FAISS indexes live here)
└── download_flickr.py        # Kagglehub dataset downloader
```

---

## 🤝 Future Enhancements
- **GPU Acceleration:** Upgrade from `faiss-cpu` to `faiss-gpu` for extreme scale indexing.
- **Dockerization:** Complete the multi-container `docker-compose` environment for one-click production deployment.
- **Approximate Nearest Neighbors (ANN):** Shift from exact search (`IndexFlatIP`) to HNSW (`IndexHNSW`) for sub-millisecond retrieval on datasets exceeding 1M+ vectors.
