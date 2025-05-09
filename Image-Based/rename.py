import os
import shutil
import pillow_heif
from PIL import Image

# Register HEIF opener
pillow_heif.register_heif_opener()

# Create or clean To_Label directory
if os.path.exists("To_Label"):
    # Remove all files in To_Label
    for file in os.listdir("To_Label"):
        file_path = os.path.join("To_Label", file)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs("To_Label")

# Convert HEIC to JPG
for filename in os.listdir("Archive"):
    if filename.endswith(".HEIC"):
        # Create new filename with .jpg extension
        new_filename = filename.replace(".HEIC", ".jpg")
        
        # Source and destination paths
        src = os.path.join("Archive", filename)
        dst = os.path.join("To_Label", new_filename)
        
        # Open and convert HEIC to JPG
        img = Image.open(src)
        img.save(dst, "JPEG")

print("Finished converting HEIC files to JPG in To_Label directory")
