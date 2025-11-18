import os
from PIL import Image

# Hardcoded path to the folder containing images
INPUT_FOLDER = "humans_faces/128"
OUTPUT_FOLDER = "humans_faces/100"

# Hardcoded new size (square)
NEW_SIZE = 120

def resize_images(input_folder, output_folder, size):
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop over all files in the folder
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        # Try to open image files only
        try:
            with Image.open(filepath) as img:
                # Convert to RGB (to avoid issues with PNG transparency, etc.)
                img = img.convert("RGB")

                # Resize image (square)
                img_resized = img.resize((size, size), Image.LANCZOS)

                # Save to output folder
                save_path = os.path.join(output_folder, filename)
                img_resized.save(save_path)

                #print(f"Resized {filename} -> {save_path}")
        except Exception as e:
            print(f"Skipped {filename}: {e}")

if __name__ == "__main__":
    resize_images(INPUT_FOLDER, OUTPUT_FOLDER, NEW_SIZE)

