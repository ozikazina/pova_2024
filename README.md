# POVa 2024 Project - Content-Based Image Search

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Create required directories:
```bash
mkdir -p models
mkdir -p images
```

## Usage

### Setting up the Dataset

1. Prepare the COCO dataset with embeddings using the chosen model:
```bash
python create_indices.py --use-coco -m CLIP
```

The `-m` flag specifies which model to use. Available options are:
- CLIP (default ViT-B/32)
- CLIP-L (ViT-L/14)
- DINOv2
- ResNet
- ViT
- DeepHash ViT
- DeepHash ResNet

### Running Benchmarks

1. Performance benchmarks:
```bash
python benchmark.py --db_size 1000000 --search_size 20000 indices/clip.npz
```

2. Memory usage analysis:
```bash
python benchmark_memory.py --db_size 1000000 indices/clip.npz
```

3. Indexing method benchmarks:
```bash
# FAISS benchmarks
python benchmark_faiss.py indices/clip.npz

# HNSW benchmarks
python benchmark_hnsw.py indices/clip.npz
```

### Running the Search Interface

Start the web interface:
```bash
python app.py
```