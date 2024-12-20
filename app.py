import torch
import gradio as gr
import numpy as np
from model import HashNet, ResNet, DeiT, ViT, AlexNet, transform_image
from pathlib import Path
import timm
import faiss
from datetime import datetime

from transformers import CLIPProcessor, CLIPModel

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

image_embeddings: faiss.Index = None
cosine_embeddings: faiss.Index = None
image_names = []


def load_indices(name):
    global image_embeddings, cosine_embeddings, image_names
    data = np.load(f"indices/{name}.npz")
    image_names = data["names"]

    embeddings = data["index"]

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.train(embeddings)
    index.add(embeddings)
    image_embeddings = index

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.train(embeddings)
    index.add(embeddings)
    cosine_embeddings = index


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
load_indices("clip")
nprobe = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def on_select(backend_type, image_input, cosine_distance):
    global model, processor
    if backend_type == "CLIP":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        load_indices("clip")
    elif backend_type == "CLIP-L":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        load_indices("clip_l")
    elif backend_type == "DINOv2":
        model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
        _ = model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        processor = timm.data.create_transform(**data_config, is_training=False)
        load_indices("dinov2")
    elif backend_type == "ResNet":
        model = HashNet(ResNet())
        model.net.load_state_dict(
            torch.load(
                "models/model_resnet.pth", weights_only=True, map_location=device
            )
        )
        load_indices("resnet")
    elif backend_type == "AlexNet":
        model = HashNet(AlexNet())
        model.net.load_state_dict(
            torch.load(
                "models/model_alexnet.pth", weights_only=True, map_location=device
            )
        )
        load_indices("alexnet")
    elif backend_type == "ViT":
        model = HashNet(ViT())
        model.net.load_state_dict(
            torch.load("models/model_vit.pth", weights_only=True, map_location=device)
        )
        load_indices("vit")
    elif backend_type == "DeiT":
        model = HashNet(DeiT())
        model.net.load_state_dict(
            torch.load("models/model_deit.pth", weights_only=True, map_location=device)
        )
        load_indices("deit")

    model.nprobe = nprobe

    if image_input:
        images = on_image_upload(image_input, backend_type, cosine_distance)
    else:
        images = []
    return (f"{backend_type} loaded.", images)


def on_set_probe(nprobe_val):
    global nprobe
    nprobe = nprobe_val
    # model.nprobe = nprobe_val


def on_image_upload(image, backend_type, cosine_distance):
    start_time = datetime.now()
    if backend_type.startswith("CLIP"):
        with torch.no_grad():
            inputs = processor(images=[image], return_tensors="pt")
            embedding: torch.Tensor = model.get_image_features(**inputs)
    elif backend_type == "DINOv2":
        with torch.no_grad():
            embedding = model(processor(image).to(device).unsqueeze(0))
    else:
        img_tensor = transform_image(image)
        with torch.no_grad():
            embedding: torch.Tensor = model(img_tensor)
    embed_time = datetime.now()

    embedding = embedding.numpy()
    if cosine_distance:
        faiss.normalize_L2(embedding)
        _, sims = cosine_embeddings.search(embedding, 10)
    else:
        _, sims = image_embeddings.search(embedding, 10)
    end_time = datetime.now()

    print(
        (embed_time - start_time).total_seconds(),
        (end_time - embed_time).total_seconds(),
    )
    return [f"images/{image_names[x]}" for x in sims[0]]


with gr.Blocks() as demo:
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload image")
        image_gallery = gr.Gallery(label="Similar images", elem_id="gallery")
    with gr.Row():
        backend_type = gr.Dropdown(
            ["CLIP", "CLIP-L", "DINOv2", "AlexNet", "ResNet", "ViT", "DeiT"], label="Model"
        )
        distance_type = gr.Checkbox(
            label="Cosine distance"
        )
        # nprobe_slider = gr.Slider(
        #     minimum=1, maximum=20, value=5, step=1, label="Probes"
        # )
        display = gr.Text(label="Info")

    image_input.upload(
        on_image_upload, inputs=[image_input, backend_type, distance_type], outputs=image_gallery
    )
    backend_type.select(on_select, [backend_type, image_input, distance_type], [display, image_gallery])
    # nprobe_slider.change(on_set_probe, nprobe_slider, None)

demo.launch()
