from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import tqdm
from coco_handler import prepare_coco

argp = ArgumentParser()
argp.add_argument(
    "-m",
    "--model",
    help="Only embed using this model: AlexNet, ResNet, ViT, DeiT, CLIP",
)
argp.add_argument(
    "-d", "--dataset", type=Path, default=Path("images"), help="Image dataset path."
)
argp.add_argument(
    "-o", "--output", type=Path, default=Path("indices"), help="Index output path."
)
argp.add_argument(
    "--use-coco", action="store_true", help="Download and use COCO dataset"
)
argp.add_argument(
    "--coco-size", type=int, default=5000, help="Number of COCO images to use"
)
argp.add_argument(
    "--dataset-type",
    type=str,
    choices=["test", "val"],
    default="val",
    help="COCO dataset type to use (test=40K images, val=5K images)",
)
args = argp.parse_args()

from model import HashNet, ResNet, DeiT, ViT, AlexNet, transform_image
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from typing import Callable, Any
import timm

if args.use_coco:
    prepare_coco(args.dataset, args.coco_size, args.dataset_type)

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_index(
    model,
    dataset_dir: Path,
    processor: Callable[[Image.Image], Any] = None,
    transform=None,
    d=64,
    num_images=5000,
):
    print("Embedding dataset.")

    image_embeddings = []
    image_names = []

    model.to(device)

    with torch.no_grad(), tqdm.tqdm(desc="Embedding", total=num_images) as pbar:
        for _, file in zip(range(num_images), dataset_dir.glob("*.jpg")):
            img = Image.open(file).convert("RGB")
            if processor is not None:
                processed = processor(img)
                processed["pixel_values"] = processed["pixel_values"].to(device)
                embedding = model.get_image_features(**processed)
            elif transform is not None:
                embedding = model(transform(img).to(device).unsqueeze(0))
            else:
                embedding = model(transform_image(img).to(device))
            image_embeddings.append(embedding.squeeze(0))
            image_names.append(file.name)
            pbar.update()

    image_embeddings = torch.stack(image_embeddings).cpu()

    return image_embeddings, image_names


def load_model(model: HashNet, weights_path: str):
    print("Loading model.")
    model.net.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=device)
    )
    model.to(device)
    return model


def write_indices(index, names, title, output_path: Path):
    print("Writing indices.")
    np.savez(str(output_path / f"{title}.npz"), index=index, names=names)


model_name = "" if args.model is None else args.model.lower()

if model_name == "" or model_name == "dh_vit":
    model = load_model(HashNet(ViT()), "models/model_vit.pth")
    index, names = create_index(model, args.dataset, num_images=args.coco_size)
    write_indices(index, names, "dh_vit", args.output)

if model_name == "" or model_name == "dh_deit":
    model = load_model(HashNet(DeiT()), "models/model_deit.pth")
    index, names = create_index(model, args.dataset, num_images=args.coco_size)
    write_indices(index, names, "dh_deit", args.output)

if model_name == "" or model_name == "dh_resnet":
    model = load_model(HashNet(ResNet()), "models/model_resnet.pth")
    index, names = create_index(model, args.dataset, num_images=args.coco_size)
    write_indices(index, names, "dh_resnet", args.output)

if model_name == "" or model_name == "dh_alexnet":
    model = load_model(HashNet(AlexNet()), "models/model_alexnet.pth")
    index, names = create_index(model, args.dataset, num_images=args.coco_size)
    write_indices(index, names, "dh_alexnet", args.output)

if model_name == "" or model_name.startswith("clip"):
    if model_name == "clip_l":
        model = timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True)
        output_name = "clip_l"
    else:
        model = timm.create_model("vit_base_patch32_clip_224.openai_ft_in1k", pretrained=True)
        output_name = "clip"

    model = model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    index, names = create_index(
        model, args.dataset, transform=transforms, num_images=args.coco_size
    )
    write_indices(index, names, model_name, args.output)

if model_name == "" or model_name == "dinov2":
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
    model = model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    index, names = create_index(
        model, args.dataset, transform=transforms, num_images=args.coco_size
    )
    write_indices(index, names, model_name, args.output)

if model_name == "" or model_name == "resnet":
    model = timm.create_model("resnet18", pretrained=True)
    model = model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    index, names = create_index(
        model, args.dataset, transform=transforms, num_images=args.coco_size
    )
    write_indices(index, names, "resnet", args.output)

if model_name == "" or model_name == "vit":
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model = model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    index, names = create_index(
        model, args.dataset, transform=transforms, num_images=args.coco_size
    )
    write_indices(index, names, "vit", args.output)

print("Done.")
