import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import json
import os


def plot_ingredient_table_grid(
    items, n_rows=None, image_list=None  # Optional: list of images, or None
):
    if n_rows is None:
        n_rows = len(items)
    max_items_per_row = [
        max(
            len(item.get("gt_ingredients", [])),
            len(item.get("pred_ingredients", [])),
            1,
        )
        for item in items
    ]
    total_height = sum(
        [max(4, n * 0.5) for n in max_items_per_row]
    )  # 0.5 per ingredient, min 4 per row

    fig, ax = plt.subplots(n_rows, 4, figsize=(20, total_height))
    columns = ["Image", "Inverse Cooking", "Retrieved", "Ground Truth"]

    if n_rows == 1:
        ax = [ax]  # Ensure ax is always a list of rows

    for row, item in enumerate(items):
        gt_ingredients = item.get("gt_ingredients", [])
        gt_indices = item.get("gt_indices", [])
        retrieval_ingredients = item.get("pred_ingredients", [])
        retrieval_indices = item.get("pred_indices", [])
        ours_ingredients = item.get("gen_ingredients_names", [])
        ours_indices = item.get("gen_indices", [])

        # 1. Image column
        ax[row][0].axis("off")
        image_path = item.get("image_path")
        if image_path:
            image_path = image_path.replace(
                "\\", "/"
            )  # Replace backslashes with forward slashes
            print("Trying to load:", image_path, "Exists:", os.path.exists(image_path))
        if image_path and os.path.exists(image_path):
            img = mpimg.imread(image_path)
            ax[row][0].imshow(img)
        else:
            ax[row][0].text(0.5, 0.5, "Image", ha="center", va="center", fontsize=12)

        # Helper to draw a single box around all text in a column
        def draw_column_box(axx):
            rect = Rectangle(
                (0.03, 0.01),
                0.94,
                0.98,
                linewidth=2,
                edgecolor="black",
                facecolor="none",
                transform=axx.transAxes,
                zorder=0,
            )
            axx.add_patch(rect)

        # Calculate the max number of items in this row (across all three columns)
        max_items = max(
            len(ours_ingredients), len(retrieval_ingredients), len(gt_ingredients), 1
        )
        step = 1.0 / (max_items + 1)
        y_start = 1.0 - step / 2

        # 2. Ours column
        ax[row][1].axis("off")
        y = y_start
        for word, idx in zip(ours_ingredients, ours_indices):
            color = "blue" if idx in gt_indices else "red"
            ax[row][1].text(
                0.5,
                y,
                word,
                ha="center",
                va="top",
                fontsize=14,
                color=color,
                transform=ax[row][1].transAxes,
            )
            y -= step
        draw_column_box(ax[row][1])
        if row == 0:
            ax[row][1].set_title(columns[1], fontsize=16, fontweight="bold")

        # 3. Retrieval column
        ax[row][2].axis("off")
        y = y_start
        for word, idx in zip(retrieval_ingredients, retrieval_indices):
            color = "blue" if idx in gt_indices else "red"
            ax[row][2].text(
                0.5,
                y,
                word,
                ha="center",
                va="top",
                fontsize=14,
                color=color,
                transform=ax[row][2].transAxes,
            )
            y -= step
        draw_column_box(ax[row][2])
        if row == 0:
            ax[row][2].set_title(columns[2], fontsize=16, fontweight="bold")

        # 4. Ground Truth column
        ax[row][3].axis("off")
        y = y_start
        for word in gt_ingredients:
            ax[row][3].text(
                0.5,
                y,
                word,
                ha="center",
                va="top",
                fontsize=14,
                color="black",
                transform=ax[row][3].transAxes,
            )
            y -= step
        draw_column_box(ax[row][3])
        if row == 0:
            ax[row][3].set_title(columns[3], fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig("all_items_grid.png", dpi=200)
    plt.show()


# Usage
JSON_path = "path to your combined JSON file including indices and ingredients list for ground truth, retrieval, and generative, with image ids and image paths"
with open(JSON_path, "r", encoding="utf-8") as f:
    data = json.load(f)

plot_ingredient_table_grid(data["results"])
