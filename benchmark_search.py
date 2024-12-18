import faiss
import numpy as np
import torch
from pathlib import Path
import time
import matplotlib.pyplot as plt
from typing import List, Tuple
import json
from datetime import datetime
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from numpy_search import NumpySearchIndex

class BenchmarkResults:
    def __init__(self):
        self.sizes: List[int] = []
        self.faiss_times: List[float] = []
        self.numpy_times: List[float] = []
        self.faiss_memory: List[float] = []
        self.numpy_memory: List[float] = []
        
    def add_result(self, size: int, faiss_time: float, numpy_time: float, 
                  faiss_mem: float, numpy_mem: float):
        self.sizes.append(size)
        self.faiss_times.append(faiss_time)
        self.numpy_times.append(numpy_time)
        self.faiss_memory.append(faiss_mem)
        self.numpy_memory.append(numpy_mem)
    
    def save_to_file(self, filename: str):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'sizes': self.sizes,
            'faiss_times': self.faiss_times,
            'numpy_times': self.numpy_times,
            'faiss_memory': self.faiss_memory,
            'numpy_memory': self.numpy_memory,
            'timestamp': datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

def get_memory_usage(obj) -> float:
    if isinstance(obj, faiss.Index):
        return obj.ntotal * obj.d * 4 / (1024 * 1024)  # FAISS memory usage in MB
    elif isinstance(obj, NumpySearchIndex):
        return obj.embeddings.nbytes / (1024 * 1024)
    return 0

def load_coco_embeddings(coco_dir: Path, model_name: str = "openai/clip-vit-base-patch32", max_images: int = None) -> Tuple[torch.Tensor, List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [f for f in coco_dir.iterdir() if f.suffix.lower() in ALLOWED_SUFFIXES]
    
    if max_images:
        image_files = image_files[:max_images]
    
    embeddings = []
    image_names = []
    
    print(f"Processing {len(image_files)} images...")
    with torch.no_grad():
        for img_path in tqdm(image_files):
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=[image], return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                embedding = model.get_image_features(**inputs)
                embeddings.append(embedding.cpu())
                image_names.append(img_path.name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    return torch.cat(embeddings), image_names

def benchmark_search_performance(embeddings: torch.Tensor, 
                               subset_sizes: List[int],
                               image_names: List[str],
                               n_queries: int = 100,
                               k: int = 10) -> BenchmarkResults:
    results = BenchmarkResults()
    d = embeddings.shape[1]  # embedding dimension
    
    query_indices = np.random.choice(len(embeddings), min(n_queries, len(embeddings)), replace=False)
    query_embeddings = embeddings[query_indices]
    
    for size in subset_sizes:
        if size > len(embeddings):
            print(f"Skipping size {size} as it exceeds dataset size {len(embeddings)}")
            continue
            
        print(f"\nBenchmarking with {size} embeddings...")
        
        subset = embeddings[:size]
        subset_np = subset.numpy()
        subset_names = image_names[:size]
        
        # Indexing
        quantizer = faiss.IndexFlatL2(d)
        if size < 1000:
            ncenters = 1
        else:
            ncenters = int(size ** 0.5)
        index = faiss.IndexIVFFlat(quantizer, d, ncenters)
        index.train(subset_np)
        index.add(subset_np)
        
        numpy_index = NumpySearchIndex(subset_np, subset_names)
        
        # Benchmark
        faiss_times = []
        for query in query_embeddings:
            start = time.perf_counter()
            index.search(query.numpy().reshape(1, -1), k)
            faiss_times.append((time.perf_counter() - start) * 1000)
    
        numpy_times = []
        for query in query_embeddings:
            start = time.perf_counter()
            numpy_index.search(query.numpy(), k)
            numpy_times.append((time.perf_counter() - start) * 1000)
        
        avg_faiss_time = sum(faiss_times) / len(faiss_times)
        avg_numpy_time = sum(numpy_times) / len(numpy_times)
        faiss_memory = get_memory_usage(index)
        numpy_memory = get_memory_usage(numpy_index)
        
        print(f"Average FAISS search time: {avg_faiss_time:.2f}ms")
        print(f"Average NumPy search time: {avg_numpy_time:.2f}ms")
        print(f"FAISS memory usage: {faiss_memory:.2f}MB")
        print(f"NumPy memory usage: {numpy_memory:.2f}MB")
        
        results.add_result(size, avg_faiss_time, avg_numpy_time, 
                         faiss_memory, numpy_memory)
    
    return results

def plot_results(results: BenchmarkResults, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results.sizes, results.faiss_times, 'b-', label='FAISS')
    plt.plot(results.sizes, results.numpy_times, 'r-', label='NumPy')
    plt.xlabel('Dataset Size')
    plt.ylabel('Average Search Time (ms)')
    plt.title('Search Time vs Dataset Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'search_times.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(results.sizes, results.faiss_memory, 'b-', label='FAISS')
    plt.plot(results.sizes, results.numpy_memory, 'r-', label='NumPy')
    plt.xlabel('Dataset Size')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Dataset Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'memory_usage.png')
    plt.close()

def main():
    coco_dir = Path("images")
    output_dir = Path("benchmark_results")
    
    subset_sizes = [100, 500, 1000, 2000, 3000, 5000]  # Change for bigger dataset size
    
    print("Loading and embedding images...")
    embeddings, image_names = load_coco_embeddings(coco_dir)
    print(f"Loaded {len(embeddings)} embeddings")
    
    results = benchmark_search_performance(embeddings, subset_sizes, image_names)
    
    results.save_to_file(output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    plot_results(results, output_dir)
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main()