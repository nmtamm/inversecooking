import os
import json

# Paths to your files
current_file = "path to your current combined JSON file"
data_folder = "path to images folder"
output_file = "path to your output combined JSON file with image paths added"

# Load current file
with open(current_file, "r", encoding="utf-8") as f:
    current_data = json.load(f)


def find_image_path(image_id, root_folder):
    for root, dirs, files in os.walk(root_folder):
        if image_id in files:
            return os.path.join(root, image_id)
    return None


# Add image_path to each item
for item in current_data["results"]:
    image_id = item.get("image_id")
    if image_id:
        # Remove "Image" prefix if present
        image_id_search = image_id[5:] if image_id.startswith("Image") else image_id
        img_path = find_image_path(image_id_search, data_folder)
        if img_path:
            item["image_path"] = img_path
        else:
            item["image_path"] = None

# Save the updated file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(current_data, f, indent=2, ensure_ascii=False)
