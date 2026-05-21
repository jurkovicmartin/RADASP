import torch
import numpy as np
import logging

from dataset.dataset import SmallDataset
from utils.visual import separate_visual, overlap_visual, mask_comparison_visual
from utils.metrics import evaluate_metrics, threshold_pred
from architecture.wrapper import ModelWrapper

from architecture.radasp import RADASP



### PARAMETERS
# Dataset
images_path = "./data/data300/img"
masks_path = "./data/data300/mask"
image_size = 224
normalization = "xrv"
threshold = 0.5 # Threshold for binary mask

# Model
saves_dir = "./saves/"
logs_dir = "./logs/"
model_name = "m90"
supervision = True
architecture = RADASP(1, 1, supervision)
# Number of trained models (cross validation)
num_models = 3
use_checkpoint = True
# Test on specific samples from the dataset (indexes start at 1)
samples = [] # Empty for full set
display = True

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)

overall_metrics = {
    "pixel_auroc": [],
    "pixel_ap": [],
    "iou": [],
    "f1": [],
    "pixel_accuracy": []
}

def main():
    ### Dataset

    dataset = SmallDataset(images_path, masks_path, image_size, normalization)
    dataset.cross_validation_split(1, False)
    fold = dataset.folds[0]
    num_samples = len(fold)

    ### Model

    models = []
    if num_models == 1:
        wrapper = ModelWrapper(architecture, model_name, saves_dir, logs_dir, device)
        if use_checkpoint:
            wrapper.load_best_checkpoint()
        else:
            wrapper.load_weights()
            
        models.append(wrapper)
    # Multiple models = cross-validation
    else:
        for i in range(1, num_models + 1):
            wrapper = ModelWrapper(architecture, f"{model_name}_{i}", saves_dir, logs_dir, device)
            if use_checkpoint:
                wrapper.load_best_checkpoint()
            else:
                wrapper.load_weights()
            models.append(wrapper)

    # If samples is empty, use all samples
    global samples
    if samples == []:
        samples = range(1, num_samples + 1)

    ### Testing

    with torch.inference_mode():
        processed_samples = 0

        for i in samples:
            sample_predictions = []
            sample_metrics = {
                "pixel_auroc": [],
                "pixel_ap": [],
                "iou": [],
                "f1": [],
                "pixel_accuracy": []
            }

            img, gt_mask = fold[i - 1]
            # Add batch dimension
            img = img.to(device).unsqueeze(0)

            for model in models:

                prediction = model(img)
                if supervision:
                    prediction = torch.stack(prediction)
                    prediction = torch.mean(prediction, dim=0)

                prediction = torch.sigmoid(prediction).detach().squeeze(0).cpu().numpy()
                sample_predictions.append(prediction)

                metrics = evaluate_metrics(prediction, gt_mask.numpy(), threshold)
                # Sample with empty mask
                if not metrics: continue

                sample_metrics["pixel_auroc"].append(metrics["pixel_auroc"])
                sample_metrics["pixel_ap"].append(metrics["pixel_ap"])
                sample_metrics["iou"].append(metrics["iou"])
                sample_metrics["f1"].append(metrics["f1"])
                sample_metrics["pixel_accuracy"].append(metrics["pixel_accuracy"])


            processed_samples += 1

            # Combine predictions from all models
            final_prediction = np.stack(sample_predictions, axis=0)
            final_prediction = np.mean(final_prediction, axis=0)
            # Empty mask
            if not sample_metrics["pixel_auroc"]:
                logging.info(f"Sample {samples[processed_samples-1]} has empty mask")
            else:
                final_metrics = {
                    "pixel_auroc": sum(sample_metrics["pixel_auroc"]) / len(sample_metrics["pixel_auroc"]),
                    "pixel_ap": sum(sample_metrics["pixel_ap"]) / len(sample_metrics["pixel_ap"]),
                    "iou": sum(sample_metrics["iou"]) / len(sample_metrics["iou"]),
                    "f1": sum(sample_metrics["f1"]) / len(sample_metrics["f1"]),
                    "pixel_accuracy": sum(sample_metrics["pixel_accuracy"]) / len(sample_metrics["pixel_accuracy"])
                }

                print(f"""
                    Sample {samples[processed_samples-1]} metrics:
                    Pixel AUROC: {final_metrics["pixel_auroc"]}
                    Pixel AP: {final_metrics["pixel_ap"]}
                    IoU: {final_metrics["iou"]}
                    F1: {final_metrics["f1"]}
                    Pixel Accuracy: {final_metrics["pixel_accuracy"]}
                    """)
                
                overall_metrics["pixel_auroc"].append(final_metrics["pixel_auroc"])
                overall_metrics["pixel_ap"].append(final_metrics["pixel_ap"])
                overall_metrics["iou"].append(final_metrics["iou"])
                overall_metrics["f1"].append(final_metrics["f1"])
                overall_metrics["pixel_accuracy"].append(final_metrics["pixel_accuracy"])
            
            ### Displaying
            if display:
                pred_mask = threshold_pred(final_prediction, threshold)
                
                imgs = [img.squeeze(0).cpu().numpy().transpose(1, 2, 0),
                        gt_mask.numpy().transpose(1, 2, 0),
                        final_prediction.transpose(1, 2, 0),
                        pred_mask.transpose(1, 2, 0)]
                
                labels = [f"Input {samples[processed_samples-1]}", "Ground-Truth mask", "Prediction map", "Prediction mask"]
                
                # separate_visual(imgs, labels)

                mask_comparison_visual(imgs[1], imgs[3], labels[1], labels[3])

    print(f"""Overall metrics:
          Pixel AUROC: {sum(overall_metrics["pixel_auroc"]) / len(overall_metrics["pixel_auroc"])}
          Pixel AP: {sum(overall_metrics["pixel_ap"]) / len(overall_metrics["pixel_ap"])}
          IoU: {sum(overall_metrics["iou"]) / len(overall_metrics["iou"])}
          F1: {sum(overall_metrics["f1"]) / len(overall_metrics["f1"])}
          Pixel Accuracy: {sum(overall_metrics["pixel_accuracy"]) / len(overall_metrics["pixel_accuracy"])}
          """)

        



if __name__ == "__main__":
    main()