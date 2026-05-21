import logging
import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from dataset.dataset import SmallDataset, Fold
from architecture.wrapper import ModelWrapper

from architecture.radasp import RADASP

### HYPERPARAMETERS

# Dataset
images_path = "./data/data_clean/img"
masks_path = "./data/data_clean/mask"
size = 224
normalization = "xrv"
# 1 folds will be used for validation and testing
# 1 = no cross validation and 1/3 of the dataset will be used for testing
folds_num = 3       # Number of folds for cross validation
# Set to 0 or None not use validation
val_ratio = 0.4     # Validation ratio (0.2 = 20% of the testing fold will be taken for validation)
batch_size = 8
# Model
saves_dir = "./saves/"
model_name = "m100"
supervision = True
architecture = RADASP(1, 1, supervision)
load_pretrained = False     # Load pretrained model from save
save_model = True           # Save model after training
use_checkpoints = True      # Ongoing best checkpoint save
# Training
epochs = 2000                # Number of training epochs
learning_rate = 1e-1   
eta_min = 1e-3              # Minimum learning rate (scheduler)
val_interval = 5           # Interval at which to validate [epochs] (also file logging interval)
patience = 750              # Patience epochs for early stopping
augmentation = True
freeze_epochs = None           # Number of epochs to freeze the backbone
# Logging interval = val_interval
logs_dir = "./logs/"
# Testing
display_batches = []        # Batches that will be displayed

device = "cuda" if torch.cuda.is_available() else "cpu"

metrics = {
    "pixel_auroc": [],
    "pixel_ap": [],
    "iou": [],
    "f1": [],
    "pixel_accuracy": []
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)

seed = 20
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    logging.info(f"Device: {device}")
    
    ### DATASET

    dataset = SmallDataset(images_path, masks_path, size, normalization)
    dataset.cross_validation_split(folds_num)
    folds = dataset.folds

    dataset_info = f"""Dataset loaded.
                 Number of samples: {len(dataset)}
                 Number of folds: {folds_num}
                 Image size: {size}x{size}
                 Normalization: {normalization}
                 Augmentation: {augmentation}
                 Supervision: {supervision}
                 Freeze epochs: {freeze_epochs}
                 Validation ratio: {val_ratio * 100}%
                 """
    
    logging.info(dataset_info)
    
    ### TRAINING and TESTING

    if folds_num == 1:
        basic_fit(folds[0], dataset_info)
    else:
        cross_validation_fit(folds, dataset_info)



def basic_fit(fold: Fold, info: str):
    """
    Train and test a model using a single fold.

    Args:
        fold (Fold): A single fold to train and test the model on.
    """
    try:
        logging.info(f"""
                        *************
                        * {model_name} *
                        *************
                        """)
        # Model
        wrapper = ModelWrapper(architecture, model_name, saves_dir, logs_dir, device)
        wrapper.file_log.log(info)

        if load_pretrained:
            wrapper.load_weights()

        # Dataloaders
        test_fold, train_fold = Fold.split_fold(fold, 0.3)
        if val_ratio is None or val_ratio == 0:
            val_loader = None
        else:
            # Training with validation
            val_fold, test_fold = Fold.split_fold(test_fold, val_ratio)
            val_loader = DataLoader(val_fold, batch_size=batch_size)

        test_loader = DataLoader(test_fold, batch_size=batch_size)

        # Set augmentation
        train_fold.augment = augmentation

        train_loader = DataLoader(train_fold, batch_size=batch_size)
        
        # Training
        wrapper.fit(train_loader, val_loader, epochs, val_interval, learning_rate, eta_min, freeze_epochs, patience)

        if save_model:
            wrapper.save_weights()

        # Testing
        # Update weights to best checkpoint
        if use_checkpoints:
            wrapper.load_best_checkpoint()

        model_metrics = wrapper.test(test_loader, display_batches)

        metrics["pixel_auroc"].append(model_metrics["pixel_auroc"])
        metrics["pixel_ap"].append(model_metrics["pixel_ap"])
        metrics["iou"].append(model_metrics["iou"])
        metrics["f1"].append(model_metrics["f1"])
        metrics["pixel_accuracy"].append(model_metrics["pixel_accuracy"])

    # Premature training exit
    except KeyboardInterrupt:
        logging.info("""
        *************************
        * Training interrupted. *
        *************************
                        """)
        if input("Do you want to save the model? (y/n): ").lower() == "y":
            wrapper.save_weights()
        return


def cross_validation_fit(folds: list[Fold], info: str):
    """
    Perform cross-validation training and testing on the given folds.

    Args:
        folds (list[Fold]): A list of folds to perform cross-validation on.
    """
    try:
        models = []
        for i in range(folds_num):
            current_name = f"{model_name}_{i + 1}"
            wrapper = ModelWrapper(architecture, current_name, saves_dir, logs_dir, device)
            models.append(wrapper)

            wrapper.file_log.log(info)

        for i, wrapper in enumerate(models):
            logging.info(f"""
                            *************
                            * Model {i + 1}  *
                            *************
                            """)
            if load_pretrained:
                wrapper.load_weights()

            # Dataloaders
            test_fold = folds[i]
            if val_ratio is None or val_ratio == 0:
                val_loader = None
            else:
                # Training with validation
                val_fold, test_fold = Fold.split_fold(test_fold, val_ratio)
                val_loader = DataLoader(val_fold, batch_size=batch_size)

            test_loader = DataLoader(test_fold, batch_size=batch_size)

            train_fold = folds[:i] + folds[i+1:]
            train_fold = Fold.combine_folds(train_fold)

            # Set augmentation
            train_fold.augment = augmentation

            train_loader = DataLoader(train_fold, batch_size=batch_size)
            
            # Training
            wrapper.fit(train_loader, val_loader, epochs, val_interval, learning_rate, eta_min, freeze_epochs, patience)

            if save_model:
                wrapper.save_weights()

            # Testing
            # Update weights to best checkpoint
            if use_checkpoints:
                wrapper.load_best_checkpoint()

            model_metrics = wrapper.test(test_loader, display_batches)

            metrics["pixel_auroc"].append(model_metrics["pixel_auroc"])
            metrics["pixel_ap"].append(model_metrics["pixel_ap"])
            metrics["iou"].append(model_metrics["iou"])
            metrics["f1"].append(model_metrics["f1"])
            metrics["pixel_accuracy"].append(model_metrics["pixel_accuracy"])


        # Testing summary
        result_mes = """
                        ***********
                        * Summary *
                        ***********
                    """
        
        for i in range(folds_num):
            result_mes += f"""
                            ***********
                            * Model {i + 1} *
                            ***********

                            Pixel AUROC: {metrics["pixel_auroc"][i]}
                            Pixel AP: {metrics["pixel_ap"][i]}
                            IoU: {metrics["iou"][i]}
                            F1: {metrics["f1"][i]}
                            Pixel Accuracy: {metrics["pixel_accuracy"][i]}
                            """
            
        result_mes += f"""
                        ***********
                        * Overall *
                        ***********

                        Pixel AUROC: {sum(metrics["pixel_auroc"]) / folds_num}
                        Pixel AP: {sum(metrics["pixel_ap"]) / folds_num}
                        IoU: {sum(metrics["iou"]) / folds_num}
                        F1: {sum(metrics["f1"]) / folds_num}
                        Pixel Accuracy: {sum(metrics["pixel_accuracy"]) / folds_num}
                        """
        
        for wrapper in models:
            wrapper.file_log.log(result_mes)

        logging.info(result_mes)

    # Premature training exit
    except KeyboardInterrupt:
        logging.info("""
        *************************
        * Training interrupted. *
        *************************
                        """)
        if input("Do you want to save the model? (y/n): ").lower() == "y":
            wrapper.save_weights()
        return
    




if __name__ == "__main__":
    main()