import os
import shutil
import glob

try:
    import kagglehub
except ImportError:
    print("Error: kagglehub is not installed. Please run: pip install -r requirements-data.txt")
    exit(1)

def main():
    print("Downloading Flickr30k dataset via kagglehub...")
    path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
    print(f"Dataset downloaded to cache: {path}")

    local_data_dir = os.path.abspath("data")
    local_images_dir = os.path.join(local_data_dir, "images")
    local_captions_file = os.path.join(local_data_dir, "captions.txt")

    print("Locating dataset components in the cache...")
    
    # 1. Locate exactly one results.csv
    csv_candidates = glob.glob(os.path.join(path, "**", "results.csv"), recursive=True)
    if len(csv_candidates) != 1:
        print(f"Error: Expected exactly one 'results.csv', found {len(csv_candidates)}.")
        return
    results_csv_path = csv_candidates[0]

    # 2. Locate exactly one image directory
    image_dir_candidates = set()
    for root, dirs, files in os.walk(path):
        if any(f.lower().endswith('.jpg') for f in files):
            image_dir_candidates.add(root)
            
    if len(image_dir_candidates) != 1:
        print(f"Error: Expected exactly one directory containing .jpg files, found {len(image_dir_candidates)}.")
        return
    images_source_dir = list(image_dir_candidates)[0]

    # 3. Validate image count
    jpg_count = sum(1 for f in os.listdir(images_source_dir) if f.lower().endswith('.jpg'))
    if jpg_count < 30000:
        print(f"Error: Found only {jpg_count} images in {images_source_dir}. Expected >30000.")
        return

    print("Validation successful. Starting staging process...")

    # 4. Stage both images and captions
    staging_dir = os.path.join(local_data_dir, "staging")
    staging_images_dir = os.path.join(staging_dir, "images")
    staging_captions_file = os.path.join(staging_dir, "captions.txt")

    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_images_dir, exist_ok=True)

    print(f"Copying results.csv to staging...")
    shutil.copy2(results_csv_path, staging_captions_file)

    print(f"Copying images to staging directory (this may take a few minutes)...")
    count = 0
    for filename in os.listdir(images_source_dir):
        if filename.lower().endswith('.jpg'):
            src = os.path.join(images_source_dir, filename)
            dst = os.path.join(staging_images_dir, filename)
            shutil.copy2(src, dst)
            count += 1
            if count % 5000 == 0:
                print(f"Staged {count} images...")

    # 5. Atomic-like replacement of actual data
    print("Staging complete. Replacing existing data...")
    if os.path.exists(local_images_dir):
        shutil.rmtree(local_images_dir)
    if os.path.exists(local_captions_file):
        os.remove(local_captions_file)
        
    os.rename(staging_images_dir, local_images_dir)
    os.rename(staging_captions_file, local_captions_file)
    
    # Cleanup empty staging root
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)

    print(f"Successfully deployed {count} images and captions.txt. Ready for ingestion!")

if __name__ == "__main__":
    main()
