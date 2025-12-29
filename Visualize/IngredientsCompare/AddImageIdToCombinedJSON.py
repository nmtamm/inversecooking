import json

# Paths to your files
current_file = "path to your current combined JSON file"
new_file = "path to your mapping JSON file"
output_file = "path to your output combined JSON file with image IDs added"

# Load current combined JSON file containing ground truth, retrieval and generative recipes
with open(current_file, "r", encoding="utf-8") as f:
    current_data = json.load(f)

# Load the JSON file containing mapping from Pair ID (for retrieval) to Image ID (for generative)
"""
[
    {
        "number": "this is the index number",
        "pair id for retrieval": "this is the pair id used in retrieval (51304 in total)",
        "ground truth recipe id": "this is the ground truth recipe id",
        "image id": "this is the image id used in generative (same as in Recipe1M dataset)"
    },
    ...
]
"""
with open(new_file, "r", encoding="utf-8") as f:
    image_data = json.load(f)

# Build a mapping from pair id to image id
pair_to_image = {item["pair id for retrieval"]: item["image id"] for item in image_data}

# Add image_id to each item in current_data if prefix matches
for item in current_data["results"]:
    prefix = item.get("prefix")
    if prefix in pair_to_image:
        item["image_id"] = pair_to_image[prefix]

# Save the updated file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(current_data, f, indent=2, ensure_ascii=False)
