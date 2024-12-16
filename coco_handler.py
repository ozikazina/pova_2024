from pathlib import Path
import requests
import zipfile
from tqdm import tqdm
import shutil

class COCOHandler:
    def __init__(self, target_dir: Path):
        self.target_dir = target_dir
        self.target_dir.mkdir(exist_ok=True)
        
    def download_file(self, url: str, dest_path: Path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

    def setup_dataset(self, num_images: int = 5000):
        temp_dir = Path("temp_coco")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            image_url = "http://images.cocodataset.org/zips/val2017.zip"
            image_zip = temp_dir / "val2017.zip"
            
            print("Downloading COCO images...")
            self.download_file(image_url, image_zip)
            
            print("Extracting files...")
            with zipfile.ZipFile(image_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            val_dir = temp_dir / "val2017"
            if val_dir.exists():
                print(f"Moving {num_images} images to {self.target_dir}...")
                image_files = list(val_dir.glob("*.jpg"))[:num_images]
                
                for img_file in tqdm(image_files, desc="Moving images"):
                    shutil.copy2(img_file, self.target_dir / img_file.name)
                
                print(f"Successfully moved {len(image_files)} images")
            
        finally: # Clean up temp files
            shutil.rmtree(temp_dir)
            
        print("COCO dataset preparation complete!")
        return len(list(self.target_dir.glob("*.jpg")))

def prepare_coco(target_dir: Path, num_images: int = 5000):
    handler = COCOHandler(target_dir)
    return handler.setup_dataset(num_images)