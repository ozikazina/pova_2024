import torch
import gradio as gr
import numpy as np
from model import HashNet, ViT, ResNet, transform_image
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("vit_base_patch32_clip_224.openai_ft_in1k", pretrained=True)
model = model.eval()
data_config = timm.data.resolve_model_data_config(model)
processor = timm.data.create_transform(**data_config, is_training=False)
model.to(device)
load_indices("clip")
nprobe = 20


def on_select(backend_type, image_input, cosine_distance):
    global model, processor
    if backend_type == "CLIP":
        model = timm.create_model("vit_base_patch32_clip_224.openai_ft_in1k", pretrained=True)
        model = model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        processor = timm.data.create_transform(**data_config, is_training=False)
        load_indices("clip")
    elif backend_type == "CLIP-L":
        model = timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True)
        model = model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        processor = timm.data.create_transform(**data_config, is_training=False)
        load_indices("clip_l")
    elif backend_type == "DINOv2":
        model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
        model = model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        processor = timm.data.create_transform(**data_config, is_training=False)
        load_indices("dinov2")
    elif backend_type == "ResNet":
        model = timm.create_model("resnet18", pretrained=True)
        model = model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        processor = timm.data.create_transform(**data_config, is_training=False)
        load_indices("resnet")
    elif backend_type == "ViT":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        model = model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        processor = timm.data.create_transform(**data_config, is_training=False)
        load_indices("vit")
    elif backend_type == "DeepHash ViT":
        model = HashNet(ViT())
        model = model.eval()
        model.net.load_state_dict(
            torch.load("models/model_vit.pth", weights_only=True, map_location=device)
        )
        load_indices("dh_vit")
    elif backend_type == "DeepHash ResNet":
        model = HashNet(ResNet())
        model = model.eval()
        model.net.load_state_dict(
            torch.load("models/model_resnet.pth", weights_only=True, map_location=device)
        )
        load_indices("dh_resnet")

    model.to(device)

    if image_input:
        images = on_image_upload(image_input, backend_type, cosine_distance)
    else:
        images = []
    return (f"{backend_type} loaded.", images)


def on_set_probe(nprobe_val):
    global nprobe
    nprobe = nprobe_val

def on_cosine(backend_type, image_input, cosine_distance):
    if image_input:
        return on_image_upload(image_input, backend_type, cosine_distance)
    else:
        return []

def on_image_upload(image, backend_type, cosine_distance):
    start_time = datetime.now()
    if backend_type == "DINOv2" or backend_type.startswith("CLIP"):
        with torch.no_grad():
            embedding = model(processor(image).to(device).unsqueeze(0))
    else:
        img_tensor = transform_image(image).to(device)
        with torch.no_grad():
            embedding: torch.Tensor = model(img_tensor)
    embed_time = datetime.now()

    embedding = embedding.cpu().numpy()
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
            ["CLIP", "CLIP-L", "DINOv2", "ResNet", "ViT", "DeepHash ViT", "DeepHash ResNet"], label="Model"
        )
        distance_type = gr.Checkbox(
            label="Cosine distance",
        )
        display = gr.Text(label="Info")

    image_input.upload(
        on_image_upload, inputs=[image_input, backend_type, distance_type], outputs=image_gallery
    )
    distance_type.select(on_cosine, [backend_type, image_input, distance_type], image_gallery)
    backend_type.select(on_select, [backend_type, image_input, distance_type], [display, image_gallery])

demo.launch()
