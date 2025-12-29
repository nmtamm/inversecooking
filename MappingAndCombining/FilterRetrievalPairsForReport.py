import json

pair = [
    6855,
    14651,
    14835,
    1243,
    2119,
    4062,
    14128,
    15581,
    36499,
    3009,
    20554,
    50048,
]


def filter_retrieval_pairs(pair):
    input_path = "path to your JSON file containing all retrieval pairs"
    output_path = "path to your output JSON file for selected retrieval pairs"

    # Load pairs from JSON file
    with open(input_path, "r") as f:
        all_pairs = json.load(f)

    # Convert to 0-based indices
    zero_based_indices = [i - 1 for i in pair]

    # Extract the specified pairs
    selected_pairs = [all_pairs[i] for i in zero_based_indices]

    # Print or save the selected pairs
    print(selected_pairs)

    # Save to a new JSON file
    with open(output_path, "w") as f:
        json.dump(selected_pairs, f, indent=4)


def add_detail_to_filtered_pairs():
    detailed_path = "path to your JSON file containing detailed retrieval pairs"
    output_path = (
        "path to your output JSON file for adding detail to selected retrieval pairs"
    )

    # Load the JSON file with detailed items
    with open(detailed_path, "r") as f:
        all_items = json.load(f)

    # Build the set of target pair ids
    target_pair_ids = {f"Pair{ipair}" for ipair in pair}

    # Filter items with matching "pair id for retrieval"
    filtered_items = [
        item
        for item in all_items
        if item.get("pair id for retrieval") in target_pair_ids
    ]

    # Save to a new JSON file
    with open(output_path, "w") as f:
        json.dump(filtered_items, f, indent=2)

    print(f"Found {len(filtered_items)} matching items.")


# Usage
# filter_retrieval_pairs(pair)
# add_detail_to_filtered_pairs()
