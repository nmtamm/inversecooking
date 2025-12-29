import os
import json


def find_and_merge(folder_path, current_json_path, output_path):
    # Load current JSON file of retrieval results, containing retrieved and ground truth recipes, ingredients list and indices
    with open(current_json_path, "r", encoding="utf-8") as f:
        current_data = json.load(f)
    items = current_data["results"]

    # Gather all JSON files of generative results in the folder
    json_files = [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.endswith(".json") and fname != os.path.basename(current_json_path)
    ]

    merged_results = []

    # For each item in current_data, find the matching generative result by comparing gt_indices
    for item in items:
        gt_indices = item.get("gt_indices", [])
        found = False
        for file_path in json_files:
            with open(file_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)
            # Try to get gt_indices from file_data (handle both list and dict structures)
            file_gt_indices = file_data.get("gt_indices") or file_data.get(
                "results", [{}]
            )[0].get("gt_indices", [])
            # Compare
            if gt_indices == file_gt_indices:
                # Copy all old attributes from item
                merged_item = dict(item)
                # Copy gen_indices and gen_ingredients_names from the matched file
                merged_item["gen_indices"] = file_data.get("gen_indices")
                merged_item["gen_ingredients_names"] = file_data.get("gen_ingredients")
                merged_results.append(merged_item)
                found = True
                break  # Stop after first match
        if not found:
            # If not found, just copy the item as is
            merged_results.append(item)

    # Write to new file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": merged_results}, f, indent=2, ensure_ascii=False)


# Usage:
folder_path = "path to your directory containing generative JSON files"
current_json_path = "path to your current retrieval JSON file"
output_path = "path to your output merged JSON file"

find_and_merge(folder_path, current_json_path, output_path)
