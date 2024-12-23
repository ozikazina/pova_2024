from argparse import ArgumentParser
import numpy as np
import faiss
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import annoy
import hnswlib
import sys

argp = ArgumentParser()
argp.add_argument("--db_size", default=100000, help="Maximum database size.")
argp.add_argument("--search_size", default=20000, help="Number of lookups.")
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


def benchmark_method(name, color, ef=30, M=48):
    print(name)
    timings = []
    sizes = []
    i = 64

    while i < len(embeddings):
        db = embeddings[:i]
        random_vectors = np.random.default_rng().random(
            (search_size, embeddings.shape[1]), dtype=np.float32
        )
        faiss.normalize_L2(random_vectors)

        setup_start = datetime.now()
        index = hnswlib.Index(space="ip", dim=db.shape[1])
        index.init_index(max_elements=db.shape[0], ef_construction=ef, M=M)
        index.add_items(db)
        setup_end = datetime.now()

        search_start = datetime.now()
        _ = index.knn_query(random_vectors, k=1)
        search_end = datetime.now()

        time_per_search = (search_end - search_start).total_seconds() / search_size
        print(i, "{0:.3g}".format(time_per_search))
        timings.append(time_per_search)
        sizes.append(i)

        i *= 2

    ax.loglog(sizes, timings, color, label=name)


benchmark_method("HNSW ef=30 M=10", "blue", 30, 10)
benchmark_method("HNSW ef=80 M=10", "cornflowerblue", 100, 10)
benchmark_method("HNSW ef=30 M=40", "red", 30, 40)
benchmark_method("HNSW ef=80 M=40", "orangered", 100, 40)
benchmark_method("HNSW ef=30 M=80", "green", 30, 80)
benchmark_method("HNSW ef=100 M=80", "lime", 100, 80)

ax.set_xlabel(f"Database size [{embeddings.shape[1]}D elements]")
ax.set_ylabel("Look-up time [s]")
ax.set_title("Database size vs. HNSW look-up time")
ax.legend()
plt.savefig(str(args.output / "hnsw.svg"))
plt.close(fig)
