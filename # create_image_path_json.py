# create_image_path_json.py

import os
import json
from tqdm import tqdm # A library for creating smart progress bars

# --- 1. CONFIGURATION ---

# The name of the main folder containing all your subfolders with images.
ROOT_FOLDER = 'f:\AIC25\data\keyframes\Keyframes_L21\L21_V001'

# The name of the JSON file you want to create.
OUTPUT_JSON_FILE = 'image_path_L22.json'

# A set of common image file extensions to look for.
# Using a set is slightly faster for checking. Add any other formats you have.
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


# --- 2. THE MAIN SCRIPT LOGIC ---

def create_image_path_file():
    """
    Scans a root directory, finds all image files, and creates a JSON file
    mapping a unique ID to each image's full path.
    """
    # This dictionary will store our final results, like {"0": "path/to/img1.jpg", ...}
    image_paths_dict = {}
    
    # This counter will be used to create the unique ID for each image.
    image_id_counter = 0

    print(f"Starting scan in root folder: '{ROOT_FOLDER}'...")

    # os.walk() is the perfect tool for this. It recursively goes through
    # every folder and file in a directory tree.
    # We wrap it in tqdm() to get a nice progress bar.
    for root, dirs, files in tqdm(list(os.walk(ROOT_FOLDER))):
        # We sort the files to ensure a consistent order every time you run the script
        for filename in sorted(files):
            # Check if the file's extension is in our list of valid image extensions.
            # We use .lower() to handle cases like .JPG or .PNG.
            if filename.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                
                # Create the full, cross-platform-compatible path to the image.
                full_path = os.path.join(root, filename)
                
                # On Windows, os.path.join uses backslashes (\). It's good practice
                # to convert them to forward slashes for consistency in web contexts.
                full_path = full_path.replace('\\', '/')
                
                # Add the entry to our dictionary. The key is the counter converted to a string.
                image_paths_dict[str(image_id_counter)] = full_path
                
                # Increment the counter for the next image.
                image_id_counter += 1

    if not image_paths_dict:
        print(f"Warning: No images found in '{ROOT_FOLDER}'. Please check the folder name and location.")
        return

    print(f"\nScan complete. Found {len(image_paths_dict)} images.")

    # --- 3. SAVE THE DICTIONARY TO A JSON FILE ---
    
    print(f"Saving the data to '{OUTPUT_JSON_FILE}'...")
    
    # 'with open' is the safe way to handle files in Python.
    # indent=4 makes the JSON file nicely formatted and easy for humans to read.
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(image_paths_dict, f, indent=4)
        
    print("--- All Done! ---")
    print(f"Your file '{OUTPUT_JSON_FILE}' has been created successfully.")


# This makes the script runnable from the command line
if __name__ == '__main__':
    create_image_path_file()