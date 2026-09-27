import os
import urllib.request
import zipfile
import shutil
import ssl

# Bypass Windows SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"
zip_path = "flickr8k.zip"
images_dir = "data/images"
os.makedirs(images_dir, exist_ok=True)

print("Downloading full Flickr8k dataset (~1GB). This will take a few minutes...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete. Extracting files...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("temp_flickr")

print("Routing images to data/images/...")
count = 0
for root, dirs, files in os.walk("temp_flickr"):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.move(os.path.join(root, file), os.path.join(images_dir, file))
            count += 1

os.remove(zip_path)
shutil.rmtree("temp_flickr")
print(f"Successfully processed {count} images. Ready for ingestion!")
