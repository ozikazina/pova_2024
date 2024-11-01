import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b4, resnet101, alexnet
from model import HashNet, Hash_func, AlexNet, transform_image
from pathlib import Path

# model = alexnet()
# model.fc = torch.nn.Identity()
model  = HashNet()
model.net.load_state_dict(torch.load("hashnet_pretrained.pth", weights_only=True))
model.eval()

image_embeddings = []
image_paths = []

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


for file in Path("images").iterdir():
    if file.suffix not in ALLOWED_SUFFIXES:
        continue
    
    img = Image.open(file).convert("RGB")
    embedding = model(img_transforms(img).unsqueeze(0))
    embedding /= torch.linalg.norm(embedding[0])
    image_embeddings.append((file, embedding))
    # image_paths.append(file)

def on_select(backend_type, hash_type):
    print(backend_type, hash_type)
    return "ABC"

def on_image_upload(image):
    img_tensor = img_transforms(image.convert("RGB")).unsqueeze(0)
    embedding:torch.Tensor = model(img_tensor)
    # embedding /= torch.linalg.norm(embedding[0])

    # sims = sorted(image_embeddings, key=lambda x: torch.dot(x[1][0], embedding[0]), reverse=True)
    sims = sorted(image_embeddings, key=lambda x: torch.linalg.norm(x[1][0] - embedding[0]), reverse=False)
    return [x[0] for x in sims][:5]

with gr.Blocks() as demo:
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload image")
        image_gallery = gr.Gallery(label="Similar images", elem_id="gallery")
    with gr.Row():
        backend_type = gr.Dropdown(["ResNet", "AlexNet", "ViT"])
        hash_type = gr.Dropdown(["Hash", "Composition"])
        display = gr.Text()

    image_input.upload(on_image_upload, inputs=image_input, outputs=image_gallery)
    backend_type.select(on_select, [backend_type, hash_type], display)

demo.launch()