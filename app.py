import torch
import gradio as gr
from model import HashNet, ResNet, DeiT, ViT, AlexNet, transform_image
from pathlib import Path
import faiss

# from torchvision.models import efficientnet_b4, resnet101, alexnet
# model = alexnet()
# model.fc = torch.nn.Identity()

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

image_embeddings:faiss.Index = None
image_names = []

def load_indices(name):
    global image_embeddings, image_names
    image_embeddings = faiss.read_index(f"indices/{name}.index")
    image_names = Path(f"indices/{name}.ids").read_text().split("\n")

model  = HashNet(AlexNet())
model.net.load_state_dict(torch.load("models/model_alexnet.pth", weights_only=True))
load_indices("vit")
nprobe = 1

def on_select(backend_type):
    global model
    if backend_type == "ResNet":
        model = HashNet(ResNet())
        model.net.load_state_dict(torch.load("models/model_resnet.pth", weights_only=True))
        load_indices("resnet")
    elif backend_type == "AlexNet":
        model = HashNet(AlexNet())
        model.net.load_state_dict(torch.load("models/model_alexnet.pth", weights_only=True))
        load_indices("alexnet")
    elif backend_type == "ViT":
        model = HashNet(ViT())
        model.net.load_state_dict(torch.load("models/model_vit.pth", weights_only=True))
        load_indices("vit")
    elif backend_type == "DeiT":
        model = HashNet(DeiT())
        model.net.load_state_dict(torch.load("models/model_deit.pth", weights_only=True))
        load_indices("deit")

    model.nprobe = nprobe
    return f"{backend_type} loaded."

def on_set_probe(nprobe_val):
    global nprobe
    nprobe = nprobe_val
    model.nprobe = nprobe_val

def on_image_upload(image):
    img_tensor = transform_image(image)
    with torch.no_grad():
        embedding:torch.Tensor = model(img_tensor)

    _, sims = image_embeddings.search(embedding, 10)
    return [f"images/{image_names[x]}" for x in sims[0]]

with gr.Blocks() as demo:
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload image")
        image_gallery = gr.Gallery(label="Similar images", elem_id="gallery")
    with gr.Row():
        backend_type = gr.Dropdown(["AlexNet", "ResNet", "ViT", "DeiT"], label="Model")
        nprobe_slider = gr.Slider(minimum=1, maximum=20, value=1, step=1, label="Probes")
        display = gr.Text(label="Info")

    image_input.upload(on_image_upload, inputs=image_input, outputs=image_gallery)
    backend_type.select(on_select, backend_type, display)
    nprobe_slider.change(on_set_probe, nprobe_slider, None)

demo.launch()