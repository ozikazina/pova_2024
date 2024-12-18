import numpy as np
from pathlib import Path
from PIL import Image
import torch
from typing import List, Tuple
import time

class NumpySearchIndex:
    def __init__(self, embeddings: np.ndarray, image_names: List[str]):
        self.embeddings = embeddings
        self.image_names = image_names
        
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
        indices = np.argsort(distances)[:k]
        
        return np.array([distances[i] for i in indices]), indices

def benchmark_search(index, query_embedding: np.ndarray, k: int = 10, n_runs: int = 100) -> float:
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        index.search(query_embedding, k)
        times.append((time.perf_counter() - start) * 1000)
    
    return sum(times) / len(times)