from argparse import ArgumentParser
from pathlib import Path
from coco_handler import prepare_coco

argp = ArgumentParser()
argp.add_argument("-m", "--model", help="Only embed using this model: AlexNet, ResNet, ViT, DeiT, CLIP")
argp.add_argument("-d", "--dataset", type=Path, default=Path("images"), help="Image dataset path.")
argp.add_argument("-o", "--output", type=Path, default=Path("indices"), help="Index output path.")
argp.add_argument("--use-coco", action="store_true", help="Download and use COCO dataset")
argp.add_argument("--coco-size", type=int, default=5000, help="Number of COCO images to use")
argp.add_argument("--dataset-type", type=str, choices=["test", "val"], default="val", help="COCO dataset type to use (test=40K images, val=5K images)")
args = argp.parse_args()

import faiss
from model import HashNet, ResNet, DeiT, ViT, AlexNet, transform_image
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from typing import Callable, Any

if args.use_coco:
    prepare_coco(args.dataset, args.coco_size, args.dataset_type)

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_index(model, dataset_dir:Path, processor:Callable[[Image.Image], Any]=None, d=64):
    print("Embedding dataset.")
    
    image_embeddings = []
    image_names = []

    with torch.no_grad():
        for file in dataset_dir.iterdir():
            if file.suffix not in ALLOWED_SUFFIXES:
                continue
            
            img = Image.open(file).convert("RGB")
            if processor is not None:
                embedding = model.get_image_features(**processor(img))
            else:
                embedding = model(transform_image(img).to(device))
            image_embeddings.append(embedding.squeeze(0))
            image_names.append(file.name)

    image_embeddings = torch.stack(image_embeddings).cpu()

    quantizer = faiss.IndexFlatL2(d)
    if image_embeddings.shape[0] < 1000:
        ncenters = 2
    else:
        ncenters = int(image_embeddings.shape[0] ** 0.5)
    index = faiss.IndexIVFFlat(quantizer, d, ncenters)
    index.train(image_embeddings)
    index.add(image_embeddings)

    return index, image_names

def load_model(model:HashNet, weights_path:str):
    print(f"Loading model.")
    model.net.load_state_dict(torch.load(weights_path, weights_only=True))
    model.to(device)
    return model

def write_indices(index, names, title, output_path:Path):
    print("Writing indices.")
    faiss.write_index(index, str(output_path / f"{title}.index"))
    output_path.joinpath(f"{title}.ids").write_text("\n".join(names))

if args.model is None or args.model.lower() == "vit":
    model = load_model(HashNet(ViT()), "models/model_vit.pth")
    index, names = create_index(model, args.dataset)
    write_indices(index, names, "vit", args.output)

if args.model is None or args.model.lower() == "deit":
    model = load_model(HashNet(DeiT()), "models/model_deit.pth")
    index, names = create_index(model, args.dataset)
    write_indices(index, names, "deit", args.output)

if args.model is None or args.model.lower() == "resnet":
    model = load_model(HashNet(ResNet()), "models/model_resnet.pth")
    index, names = create_index(model, args.dataset)
    write_indices(index, names, "resnet", args.output)

if args.model is None or args.model.lower() == "alexnet":
    model = load_model(HashNet(AlexNet()), "models/model_alexnet.pth")
    index, names = create_index(model, args.dataset)
    write_indices(index, names, "alexnet", args.output)

if args.model is None or args.model.lower() == "clip":
    model_name = "openai/clip-vit-base-patch32"  # You can choose other variants like vit-large
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    index, names = create_index(model, args.dataset, lambda img:processor(images=[img], return_tensors="pt"), d=512)
    write_indices(index, names, "clip", args.output)

print("Done.")
