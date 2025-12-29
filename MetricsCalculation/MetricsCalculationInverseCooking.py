import json
import torch
import os
import pickle
from src.utils.metrics import softIoU
from src.utils.metrics import update_error_types, compute_metrics
from rouge_score import rouge_scorer
import sacrebleu
import zipfile


# Step 1: Map the ingredient of ground truth to indices using 15k vocabulary file
def get_full_ingr_indices_from_vocab(gt_ingrs, ingr_synonym_vocab_path):
    """
    gt_ingrs: list of ground truth ingredient names (strings)
    ingr_synonym_vocab_path: path to the new JSON vocab file
    Returns: list of indices (as int) corresponding to gt_ingrs
    """
    with open(ingr_synonym_vocab_path, "r", encoding="utf-8") as f:
        ingr_vocab = json.load(f)

    # Build a mapping from synonym to index
    synonym_to_index = {}
    for idx, v in ingr_vocab.items():
        if isinstance(v, list):
            for synonym in v:
                synonym_to_index[synonym] = int(idx)
        elif isinstance(v, str):
            synonym_to_index[v] = int(idx)

    # Find indices for ground truth ingredients
    gt_ingr_pairs = []
    for ingr in gt_ingrs:
        if ingr in synonym_to_index:
            gt_ingr_pairs.append((ingr, synonym_to_index[ingr]))
    return gt_ingr_pairs


def indices_to_brief_names(indices, brief_ingr_vocab_path):
    with open(brief_ingr_vocab_path, "r", encoding="utf-8") as f:
        brief_vocab = json.load(f)  # brief_vocab should be a list of ingredient names
    return [brief_vocab[idx] for idx in indices if idx < len(brief_vocab)]


def get_pred_ingr_indices_from_vocab(ingrs, ingr_vocab_path):
    """
    ingrs: list of ingredient names (strings)
    ingr_vocab_path: path to the JSON vocab file (list of ingredient names)
    Returns: list of indices (as int) corresponding to ingrs
    """
    with open(ingr_vocab_path, "r", encoding="utf-8") as f:
        ingr_vocab = json.load(f)
    return [(ingr, ingr_vocab.index(ingr)) for ingr in ingrs if ingr in ingr_vocab]


# Step 2: Utilities for label processing and metrics computation
use_gpu = False
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
map_loc = None if torch.cuda.is_available() and use_gpu else "cpu"


# Function to convert label indices to one-hot encoded vectors
def label2onehot(labels, pad_value):
    inp_ = torch.unsqueeze(labels, 2)
    one_hot = (
        torch.FloatTensor(labels.size(0), labels.size(1), pad_value + 1)
        .zero_()
        .to(device)
    )
    one_hot.scatter_(2, inp_, 1)
    one_hot, _ = one_hot.max(dim=1)
    one_hot = one_hot[:, 1:-1]
    one_hot[:, 0] = 0
    return one_hot


# Step 3: Calculate IoU and F1 metric:
def evaluate_ingredient_pairs(data_folder, full_ingr_path, brief_ingr_path, vocab_size):
    # List all files in the data folder
    files = os.listdir(data_folder)
    # Find all ground truth files
    gt_files = [f for f in files if f.endswith("_GroundTruth.json")]
    results = []
    predicted_instr_list = []
    ground_truth_instr_list = []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for gt_file in gt_files:
        # Find the corresponding predicted file
        prefix = gt_file.replace("_GroundTruth.json", "")
        pred_file = prefix + "_Predicted.json"
        gt_path = os.path.join(data_folder, gt_file)
        pred_path = os.path.join(data_folder, pred_file)
        if not os.path.exists(pred_path):
            continue

        # Load ground truth and predicted files
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_json = json.load(f)
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_json = json.load(f)

        # Extract ingredients and instructions
        gt_ingrs = gt_json["ingredients"]
        gt_instrs = gt_json.get("instructions", [])

        # Extract all recipes from predicted file
        recipes = pred_json.get("recipes", [])
        for idx, recipe in enumerate(recipes):
            pred_ingrs = recipe.get("ingredients", [])
            pred_instrs = recipe.get("instructions", [])

            # Convert to indices
            gt_ingr_pairs = get_full_ingr_indices_from_vocab(gt_ingrs, full_ingr_path)
            pred_ingr_pairs = get_pred_ingr_indices_from_vocab(
                pred_ingrs, brief_ingr_path
            )

            gt_indices = [idx for _, idx in gt_ingr_pairs]
            pred_indices = [idx for _, idx in pred_ingr_pairs]
            gt_ingredients_names = indices_to_brief_names(gt_indices, brief_ingr_path)
            pred_ingredients_names = [name for name, _ in pred_ingr_pairs]

            if not gt_indices or not pred_indices:
                continue  # Skip this pair if either is empty

            # Convert to tensors for one-hot encoding
            gt_tensor = torch.tensor([gt_indices])
            pred_tensor = torch.tensor([pred_indices])

            # One-hot encode
            gt_one_hot = label2onehot(gt_tensor, vocab_size - 1)
            pred_one_hot = label2onehot(pred_tensor, vocab_size - 1)

            # Find matching ingredient indices
            matching_indices = set(gt_indices).intersection(set(pred_indices))

            # Calculate IoU
            iou = torch.mean(softIoU(pred_one_hot, gt_one_hot)).item()

            # Calculate F1 using compute_metrics
            error_types = {
                "tp_i": 0,
                "fp_i": 0,
                "fn_i": 0,
                "tn_i": 0,
                "tp_all": 0,
                "fp_all": 0,
                "fn_all": 0,
            }
            ret_metrics = {
                "accuracy": [],
                "f1": [],
                "jaccard": [],
                "f1_ingredients": [],
            }
            update_error_types(error_types, pred_one_hot, gt_one_hot)
            compute_metrics(ret_metrics, error_types, ["f1"])

            # Collect instructions for ROUGE/SacreBLEU
            if pred_instrs and gt_instrs:
                pred_instr_str = " ".join(pred_instrs)
                gt_instr_str = " ".join(gt_instrs)
                predicted_instr_list.append(pred_instr_str)
                ground_truth_instr_list.append(gt_instr_str)

                # Calculate ROUGE-L for this recipe
                rouge_l = scorer.score(gt_instr_str, pred_instr_str)["rougeL"].fmeasure

                # Calculate SacreBLEU for this recipe
                bleu = sacrebleu.sentence_bleu(pred_instr_str, [gt_instr_str]).score
            else:
                rouge_l = None
                bleu = None

            results.append(
                {
                    "prefix": prefix,
                    "pair": f"{prefix}_recipe{idx}",
                    "iou": iou,
                    "f1": ret_metrics["f1"][0] if ret_metrics["f1"] else None,
                    "rougeL": rouge_l,
                    "sacrebleu": bleu,
                }
            )

    return (
        results,
        predicted_instr_list,
        ground_truth_instr_list,
        gt_ingredients_names,
        pred_ingredients_names,
        gt_indices,
        pred_indices,
    )


root_dir = os.path.dirname(os.path.abspath(__file__))
vocab_path = os.path.join(root_dir, "data", "ingr_vocab.pkl")
data_folder = ""

with open(vocab_path, "rb") as f:
    ingr_vocab = pickle.load(f)

full_vocab_path = "your path to recipe1m_vocab_ingrs.json"
brief_vocab_path = os.path.join(root_dir, "data", "ingr_vocab.json")


def calculate_metrics_from_zip(dir):
    for zipfile_name in os.listdir(dir):
        # Extract zip files
        if zipfile_name.endswith(".zip"):
            print(f"Processing {zipfile_name}...")
            zip_path = os.path.join(dir, zipfile_name)
            extract_path = os.path.join(dir, zipfile_name.replace(".zip", ""))
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_path)
            except zipfile.BadZipFile:
                print(f"Warning: {zipfile_name} is not a valid zip file. Skipping.")
                continue
            data_folder = extract_path

            # Evaluate ingredient pairs
            (
                result,
                predicted_instr_list,
                ground_truth_instr_list,
                gt_ingredients_names,
                pred_ingredients_names,
                gt_indices,
                pred_indices,
            ) = evaluate_ingredient_pairs(
                data_folder=data_folder,
                full_ingr_path=full_vocab_path,
                brief_ingr_path=brief_vocab_path,
                vocab_size=len(ingr_vocab),
            )

            # Calculate averages
            ious = [r["iou"] for r in result if r["iou"] is not None]
            f1s = [r["f1"] for r in result if r["f1"] is not None]
            avg_iou = sum(ious) / len(ious) if ious else 0
            avg_f1 = sum(f1s) / len(f1s) if f1s else 0

            # Calculate ROUGE-L
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_l_scores = [
                scorer.score(gt, pred)["rougeL"].fmeasure
                for pred, gt in zip(predicted_instr_list, ground_truth_instr_list)
            ]
            avg_rouge_l = (
                sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0
            )

            # Calculate SacreBLEU
            if predicted_instr_list and ground_truth_instr_list:
                bleu = sacrebleu.corpus_bleu(
                    predicted_instr_list, [ground_truth_instr_list]
                )
                avg_bleu = bleu.score
            else:
                avg_bleu = 0

            # Define output path
            output_path = os.path.join(dir, zipfile_name.replace(".zip", ".json"))

            # Prepare the output dictionary
            output_data = {
                "average_metrics": {
                    "Average IoU": avg_iou,
                    "Average F1": avg_f1,
                    "Average ROUGE-L": avg_rouge_l,
                    "Average SacreBLEU": avg_bleu,
                },
                "results": result,
                "gt_ingredients": gt_ingredients_names,
                "gen_ingredients": pred_ingredients_names,
                "gt_indices": gt_indices,
                "gen_indices": pred_indices,
            }

            # Write to JSON file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)


def calculate_metrics_from_folders(dir):
    for folder_name in os.listdir(dir):
        folder_path = os.path.join(dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip files, only process directories

        print(f"Processing {folder_name}...")

        # Evaluate ingredient pairs
        (
            result,
            predicted_instr_list,
            ground_truth_instr_list,
            gt_ingredients_names,
            pred_ingredients_names,
            gt_indices,
            pred_indices,
        ) = evaluate_ingredient_pairs(
            data_folder=folder_path,
            full_ingr_path=full_vocab_path,
            brief_ingr_path=brief_vocab_path,
            vocab_size=len(ingr_vocab),
        )

        # Calculate averages
        ious = [r["iou"] for r in result if r["iou"] is not None]
        f1s = [r["f1"] for r in result if r["f1"] is not None]
        avg_iou = sum(ious) / len(ious) if ious else 0
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0

        # Calculate ROUGE-L
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l_scores = [
            scorer.score(gt, pred)["rougeL"].fmeasure
            for pred, gt in zip(predicted_instr_list, ground_truth_instr_list)
        ]
        avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0

        # Calculate SacreBLEU
        if predicted_instr_list and ground_truth_instr_list:
            bleu = sacrebleu.corpus_bleu(
                predicted_instr_list, [ground_truth_instr_list]
            )
            avg_bleu = bleu.score
        else:
            avg_bleu = 0

        # Define output path
        output_path = os.path.join(dir, folder_name + ".json")

        # Prepare the output dictionary
        output_data = {
            "average_metrics": {
                "Average IoU": avg_iou,
                "Average F1": avg_f1,
                "Average ROUGE-L": avg_rouge_l,
                "Average SacreBLEU": avg_bleu,
            },
            "results": result,
            "gt_ingredients": gt_ingredients_names,
            "gen_ingredients": pred_ingredients_names,
            "gt_indices": gt_indices,
            "gen_indices": pred_indices,
        }

        # Write to JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


# Usage:
# dir = "path to your directory containing zip files or folders"
# calculate_metrics_from_zip(dir)
# or
# calculate_metrics_from_folders(dir)
