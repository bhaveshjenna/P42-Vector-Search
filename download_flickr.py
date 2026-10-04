import os
import shutil
try:
    import kagglehub
except ImportError:
    print("Error: kagglehub is not installed. Please run: pip install kagglehub")
    exit(1)

def main():
    print("Downloading Flickr30k dataset via kagglehub...")
    # This downloads and extracts the dataset, returning the cache path
    path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
    print(f"Dataset downloaded to cache: {path}")

    local_images_dir = os.path.abspath("data/images")
    local_data_dir = os.path.abspath("data")
    local_captions_file = os.path.join(local_data_dir, "captions.txt")

    # Wipe existing directory to be safe (if it wasn't done manually)
    if os.path.exists(local_images_dir):
        shutil.rmtree(local_images_dir)
    os.makedirs(local_images_dir, exist_ok=True)

    print("Locating dataset components in the cache...")
    images_source_dir = None
    results_csv_path = None

    for root, dirs, files in os.walk(path):
        if "results.csv" in files:
            results_csv_path = os.path.join(root, "results.csv")
        
        # Look for a directory that contains lots of .jpg files
        jpg_count = sum(1 for f in files if f.lower().endswith('.jpg'))
        if jpg_count > 1000:  # Heuristic to find the main images folder
            images_source_dir = root
    
    if not images_source_dir or not results_csv_path:
        print("Error: Could not locate images or results.csv in the downloaded dataset.")
        return

    print(f"Copying results.csv to {local_captions_file}...")
    shutil.copy2(results_csv_path, local_captions_file)

    print(f"Copying images to {local_images_dir} (this may take a few minutes)...")
    count = 0
    for filename in os.listdir(images_source_dir):
        if filename.lower().endswith('.jpg'):
            src = os.path.join(images_source_dir, filename)
            dst = os.path.join(local_images_dir, filename)
            shutil.copy2(src, dst)
            count += 1
            if count % 5000 == 0:
                print(f"Copied {count} images...")

    print(f"Successfully copied {count} images and captions.txt. Ready for ingestion!")

if __name__ == "__main__":
    main()
