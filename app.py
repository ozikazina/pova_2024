import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b4, resnet101, alexnet
from pathlib import Path

model = alexnet()
model.fc = torch.nn.Identity()
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

    image_input.upload(on_image_upload, inputs=image_input, outputs=image_gallery)

demo.launch()