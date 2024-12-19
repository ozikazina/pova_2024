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
argp.add_argument("--db_size", default=1000000, help="Maximum database size.")
argp.add_argument("--search_size", default=20000, help="Number of lookups.")
argp.add_argument(
    "--output", type=Path, default=Path("benchmark_results"), help="Output folder."
)
argp.add_argument("--patience", type=float, default=60, help="Time limit for lookups.")
argp.add_argument(
    "--setup_patience", type=float, default=60, help="Time limit for model setup."
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


def benchmark_method(name, init, search):
    print(name)
    timings = []
    sizes = []
    i = 64
    while i < len(embeddings):
        random_vectors = np.random.default_rng().random(
            (search_size, embeddings.shape[1]), dtype=np.float32
        )

        faiss.normalize_L2(random_vectors)

        setup_start = datetime.now()
        model_data = init(embeddings[:i, :])
        setup_end = datetime.now()

        search_start = datetime.now()
        _ = search(model_data, embeddings[:i, :], random_vectors)
        search_end = datetime.now()

        time_per_search = (search_end - search_start).total_seconds() / search_size
        print(i, "{0:.3g}".format(time_per_search))
        timings.append(time_per_search)
        sizes.append(i)

        if time_per_search * search_size > args.patience:
            break
        if (setup_end - setup_start).total_seconds() > args.setup_patience:
            break
        i *= 2

    ax.loglog(sizes, timings, label=name)


benchmark_method(
    "NumPy", lambda db: None, lambda setup, db, x: np.argmax(db @ x.T, axis=0)
)


def setup_faiss_quantized(db):
    quantizer = faiss.IndexFlatIP(db.shape[1])
    ncenters = int(db.shape[0] ** 0.5)
    index = faiss.IndexIVFFlat(quantizer, db.shape[1], ncenters)
    index.train(db)
    index.add(db)
    index.nprobe = 1
    return index


benchmark_method(
    "FAISS Quantized", setup_faiss_quantized, lambda setup, db, x: setup.search(x, 1)
)


def setup_faiss(db):
    index = faiss.IndexFlatIP(db.shape[1])
    index.train(db)
    index.add(db)
    return index


benchmark_method("FAISS", setup_faiss, lambda setup, db, x: setup.search(x, 1))


def run_annoy(setup, db, x):
    for vec in x:
        _ = setup.get_nns_by_vector(vec, 1, include_distances=False)


benchmark_method("Annoy", lambda db: annoy.AnnoyIndex(db.shape[1], "dot"), run_annoy)


def setup_hnsw(db):
    index = hnswlib.Index(space="ip", dim=db.shape[1])
    index.init_index(max_elements=db.shape[0], ef_construction=30, M=48)
    index.add_items(db)
    return index


benchmark_method("HNSW", setup_hnsw, lambda setup, db, x: setup.knn_query(x, k=1))

ax.set_xlabel(f"Database size [{embeddings.shape[1]}D elements]")
ax.set_ylabel("Look-up time [s]")
ax.set_title("Database size vs. look-up time")
ax.legend()
plt.savefig(str(args.output / "results.svg"))
plt.close(fig)
