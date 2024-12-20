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
        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as file, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

    def setup_dataset(self, dataset_type: str = "val", num_images: int = 5000):
        if self.target_dir.exists() and any(self.target_dir.iterdir()):
            print("Skipping COCO download.")
            return len(list(self.target_dir.glob("*.jpg")))

        temp_dir = Path("temp_coco")
        temp_dir.mkdir(exist_ok=True)

        try:
            if dataset_type == "test":
                image_url = "http://images.cocodataset.org/zips/test2017.zip"
                subfolder = "test2017"
            else:
                image_url = "http://images.cocodataset.org/zips/val2017.zip"
                subfolder = "val2017"

            image_zip = temp_dir / f"{subfolder}.zip"

            print(f"Downloading COCO {dataset_type} images...")
            self.download_file(image_url, image_zip)

            print("Extracting files...")
            with zipfile.ZipFile(image_zip, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            dataset_dir = temp_dir / subfolder
            if dataset_dir.exists():
                print(f"Moving {num_images} images to {self.target_dir}...")
                image_files = list(dataset_dir.glob("*.jpg"))[:num_images]

                for file in self.target_dir.glob("*.jpg"):
                    file.unlink()

                chunk_size = 1000
                for i in range(0, len(image_files), chunk_size):
                    chunk = image_files[i : i + chunk_size]
                    for img_file in tqdm(
                        chunk, desc=f"Moving images batch {i//chunk_size + 1}"
                    ):
                        try:
                            shutil.copy2(img_file, self.target_dir / img_file.name)
                        except OSError as e:
                            print(f"Error copying {img_file}: {e}")
                            for temp_file in dataset_dir.glob("*.jpg"):
                                if temp_file not in image_files[: i + len(chunk)]:
                                    temp_file.unlink()
                            shutil.copy2(img_file, self.target_dir / img_file.name)

                print(f"Successfully moved {len(image_files)} images")

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

        print("COCO dataset preparation complete!")
        return len(list(self.target_dir.glob("*.jpg")))


def prepare_coco(target_dir: Path, num_images: int = 5000, dataset_type: str = "val"):
    handler = COCOHandler(target_dir)
    return handler.setup_dataset(dataset_type, num_images)
