import numpy as np
import faiss
import sys
import psutil
import gc
import os
from argparse import ArgumentParser
from pathlib import Path
import annoy
import matplotlib.pyplot as plt
import hnswlib

argp = ArgumentParser()
argp.add_argument("--db_size", default=1000000, type=int, help="Maximum database size.")
argp.add_argument("--search_size", default=20000, type=int, help="Number of lookups.")
argp.add_argument(
    "--output", type=Path, default=Path("benchmark_results"), help="Output folder."
)
argp.add_argument("index", type=Path, help=".npz embedding file path.")
args = argp.parse_args()

N_embeddings = args.db_size

try:
    data = np.load(args.index)
except Exception as e:
    print("Failed to open embedding file:", e, file=sys.stderr)
    exit(-1)

embeddings = data["index"]
embeddings = np.repeat(embeddings, N_embeddings // len(embeddings), axis=0)

embeddings = embeddings + 2 * np.random.default_rng().standard_normal(
    (N_embeddings, embeddings.shape[1]), dtype=np.float32
)

faiss.normalize_L2(embeddings)

search_size = args.search_size

fig, ax = plt.subplots()

process = psutil.Process(os.getpid())


def benchmark_method(name, color, limit: int, init):
    print(name)
    mems = []
    sizes = []
    i = 4096
    limit = min(limit, len(embeddings)) if limit > 0 else len(embeddings)

    while i < limit:
        gc.collect()

        before = process.memory_info().rss
        model = init(embeddings[:i, :])
        after = process.memory_info().rss

        setup_mem = (after - before) / (1024 * 1024)
        print(i, setup_mem)
        mems.append(setup_mem)
        sizes.append(i)

        i *= 2

        del model

    ax.loglog(sizes, mems, color, label=name)


def setup_faiss(db):
    index = faiss.IndexFlatIP(db.shape[1])
    index.train(db)
    index.add(db)
    return index


def setup_faiss_quantized(db):
    quantizer = faiss.IndexFlatIP(db.shape[1])
    ncenters = int(db.shape[0] ** 0.5)
    index = faiss.IndexIVFFlat(quantizer, db.shape[1], ncenters)
    index.train(db)
    index.add(db)
    return index


def setup_annoy(db, n_trees):
    index = annoy.AnnoyIndex(db.shape[1], "dot")
    for i, vec in enumerate(db):
        index.add_item(i, vec)

    index.build(n_trees)

    return index


def setup_hnsw(db):
    index = hnswlib.Index(space="ip", dim=db.shape[1])
    index.init_index(max_elements=db.shape[0], ef_construction=30, M=48)
    index.add_items(db)
    return index


search_vector = np.random.default_rng().random(
    (search_size, embeddings.shape[1]), dtype=np.float32
)

# benchmark_method("NumPy", "green", 200000, lambda db: search_vector @ db.T)

benchmark_method("FAISS", "blue", 0, setup_faiss)

benchmark_method("FAISS Clustered", "cornflowerblue", 0, setup_faiss_quantized)

benchmark_method("Annoy 10", "orangered", 0, lambda db: setup_annoy(db, 10))
benchmark_method("Annoy 50", "orange", 0, lambda db: setup_annoy(db, 50))

benchmark_method("HNSW", "red", 200000, setup_hnsw)

ax.set_xlabel(f"Database size [{embeddings.shape[1]}D elements]")
ax.set_ylabel("Memory [MB]")
ax.set_title("Database size vs. Memory requirements")
ax.legend()

args.output.mkdir(exist_ok=True, parents=True)

plt.savefig(str(args.output / "memory.svg"))
plt.close(fig)
