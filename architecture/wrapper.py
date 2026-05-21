import logging
from tqdm import tqdm
import os
from abc import ABC
import math
import torch
import torch.nn as nn

from utils.visual import separate_visual, save_metric_plot
from utils.metrics import evaluate_metrics, threshold_pred
from utils.logger import FileLogger
from utils.loss import FocalDiceLoss


class ModelWrapper(torch.nn.Module, ABC):
    """
    Wrapper tha provides core functionalities for models handling.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 model_name: str,
                 saves_dir: str ="./saves/",
                 logs_dir: str ="./logs/",
                 device: str ="cpu"):
        """
        Initializes the ModelWrapper.

        Args:
            model (torch.nn.Module): The neural network model to wrap.
            model_name (str): Name of the model for logging and saving.
            saves_dir (str, optional): Directory to save model weights and checkpoints. Defaults to "./saves/".
            logs_dir (str, optional): Directory to save logs and plots. Defaults to "./logs/".
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to "cpu".
        """
        super().__init__()

        self.model = model
        self.model_name = model_name
        self.saves_dir = saves_dir
        self.logs_dir = logs_dir
        self.device = device
        # Supervision determines if the model return 1 or 4 values from forward pass
        self.supervision = self.model.deep_supervision

        os.makedirs(logs_dir, exist_ok=True)
        self.file_log = FileLogger(os.path.join(logs_dir, f"{model_name}.txt"))

        os.makedirs(saves_dir, exist_ok=True)
        self.model_path = os.path.join(saves_dir, f"{model_name}")

        checkpoint_dir = os.path.join(saves_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_checkpoint_name = os.path.join(checkpoint_dir, f"{model_name}_best.pth")

        ### PARAMETERS
        self.loss_function = FocalDiceLoss(0.5)
        self.backbone_freeze = False
        # Threshold for binary masks
        self.threshold = 0.5
        self.train_metrics = {
            "loss": [],
            "pixel_auroc": [],
            "pixel_ap": [],
            "iou": [],
            "f1": [],
            "pixel_accuracy": []
        }
        # Best metrics tracker for checkpoints
        self.best_f1 = 0.0


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)
    

    def save_weights(self):
        if not self.model:
            raise ValueError("Wrapper is empty.")

        torch.save(self.model.state_dict(), self.model_path)

        logging.info(f"Model saved as {self.model_path}.")


    def load_weights(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model {self.model_path} not found.")
        
        if not self.model:
            raise ValueError("Wrapper is empty.")

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        self.model.eval()

        logging.info(f"Model {self.model_path} successfully loaded to {self.device} device.")


    def load_best_checkpoint(self):
        if not os.path.exists(self.best_checkpoint_name):
            raise FileNotFoundError(f"Checkpoint {self.best_checkpoint_name} not found.")

        checkpoint = torch.load(self.best_checkpoint_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logging.info(f"Checkpoint {self.best_checkpoint_name} successfully loaded to {self.device} device.")
        

    def fit(self,
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader =None,
              epochs: int =1,
              val_interval: int =1,
              learning_rate: float =1e-3,
              eta_min: float =1e-5,
              freeze_epochs: int = None,
              patience: int = 10
              ):
        """
        Trains the model using the provided data loaders and hyperparameters.

        Args:
            train_loader (torch.utils.data.DataLoader): Loader for training data.
            val_loader (torch.utils.data.DataLoader, optional): Loader for validation data. Defaults to None.
            epochs (int, optional): Total number of training epochs. Defaults to 1.
            val_interval (int, optional): Frequency of validation in epochs. Defaults to 1.
            learning_rate (float, optional): Base learning rate for the decoder. Defaults to 1e-3.
            eta_min (float, optional): Minimum learning rate for the scheduler. Defaults to 1e-5.
            freeze_epochs (int, optional): Number of initial epochs where the backbone is frozen. Defaults to None.
            patience (int, optional): Number of validation intervals to wait for improvement before early stopping. Defaults to 10.
        """
        if not self.model:
            raise ValueError("BaseModel wrapper is empty.")
        
        self.optimizer = torch.optim.AdamW([
            # Encoder (backbone) has 100x lower learning rate
            {"params": self.model.encoder.parameters(), "lr": learning_rate * 0.01, "name": "encoder"},
            {"params": self.model.decoder.parameters(), "lr": learning_rate, "name": "decoder"}
        ], weight_decay=1e-4)
        iterations = epochs * len(train_loader)

        # To maintain the learning rate ratio between the encoder and decoder
        def absolute_cosine_decay(step: int) -> float:
            """
            Returns a multiplier from 1.0 down to a minimum ratio.
            It acts as a uniform multiplier = the ratio between parameter groups is preserved.
            """
            progress = step / iterations
            # Convert eta_min into a ratio
            min_lr_ratio = eta_min / learning_rate 
            # Standard Cosine Annealing Math
            multiplier = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
            return multiplier

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, absolute_cosine_decay)

        msg = f"""Begin training
                     Architecture: {type(self.model)}
                     Model name: {self.model_name}
                     Epochs: {epochs}
                     Patience: {patience}
                     Batch size: {train_loader.batch_size}
                     Training set size: {len(train_loader.dataset)}
                     Learning rate: {learning_rate}
                     Eta min: {eta_min}
                     Validation set size: {len(val_loader.dataset)}
                     Validation interval: {val_interval}
                     Loss function: {self.loss_function}
                     """
        logging.info(msg)
        self.file_log.log(msg)

        self.model.to(self.device)
        
        if freeze_epochs and freeze_epochs > 0:
            self.backbone_freeze = True

            msg = f"Freezing backbone for {freeze_epochs} epochs"
            logging.info(msg)
            self.file_log.log(msg)

        self._initialize_decoder_weights()

        checkpoints_msg = ""
        last_checkpoint_change = 0

        for epoch in tqdm(range(1, epochs+1), desc="Training"):
            self.model.train()

            if epoch == freeze_epochs:
                self.backbone_freeze = False

                msg = f"Epocoh {epoch}: Unfreezing backbone"
                logging.info(msg)
                self.file_log.log(msg)
            
            self._freeze_backbone(self.backbone_freeze)            

            train_loss = 0.0
            
            for images, masks in train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                if self.supervision:
                    supervision_loss = [self.loss_function(out, masks) for out in outputs]
                    loss = sum(supervision_loss) / len(supervision_loss)
                else:
                    loss = self.loss_function(outputs, masks)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            scheduler.step()
            
            logging.info(f"Epoch {epoch}, Loss: {train_loss / len(train_loader)}")

            self.train_metrics["loss"].append(train_loss / len(train_loader))

            ### VALIDATION

            if not val_loader: continue
             
            if (epoch) % val_interval == 0:
                self.model.eval()

                metrics = self._eval_with_metrics(val_loader)

                self.train_metrics["pixel_auroc"].append(metrics["pixel_auroc"])
                self.train_metrics["pixel_ap"].append(metrics["pixel_ap"])
                self.train_metrics["iou"].append(metrics["iou"])
                self.train_metrics["f1"].append(metrics["f1"])
                self.train_metrics["pixel_accuracy"].append(metrics["pixel_accuracy"])
                
                msg = f"""Validation
                        Pixel AUROC: {metrics["pixel_auroc"]}
                        Pixel AP: {metrics["pixel_ap"]}
                        IoU: {metrics["iou"]}
                        F1: {metrics["f1"]}
                        Pixel Accuracy: {metrics["pixel_accuracy"]}
                        """

                logging.info(msg)
                self.file_log.log(f"Epoch {epoch}, Loss: {train_loss / len(train_loader)}" + msg)

                # Checkpoints
                if metrics["f1"] > self.best_f1:
                    self.best_f1 = metrics["f1"]
                    self._save_checkpoint(epoch)

                    checkpoints_msg += f"New best checkpoint saved. (Epoch: {epoch}, F1: {self.best_f1})\n"
                    last_checkpoint_change = epoch

                # Early stopping
                if epoch - last_checkpoint_change >= patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    self.file_log.log(f"Early stopping at epoch {epoch}")
                    break

        logging.info(checkpoints_msg)
        self.file_log.log(checkpoints_msg)

        self._export_metrics_plots(epoch, val_interval)

        self.model.eval()


    def test(self,
             dataloader: torch.utils.data.DataLoader,
             display_batches: list[int] = None,
             ) -> dict[str, float]:
        """
        Evaluates the model on a test dataset and logs the performance metrics.

        Args:
            dataloader (torch.utils.data.DataLoader): Loader for the test data.
            display_batches (list[int], optional): List of batch indices to visualize during testing. Defaults to None.

        Returns:
            dict[str, float]: A dictionary containing the calculated evaluation metrics.
        """
        if not self.model:
            raise ValueError("Wrapper is empty.")
        
        msg = f"""Begin testing
                     Architecture: {type(self.model)}
                     Model name: {self.model_name}
                     Batch size: {dataloader.batch_size}
                     Testing set size: {len(dataloader.dataset)}
                     Number of batches {len(dataloader)}
                     Batches that will be displayed: {display_batches}
                     """
        logging.info(msg)
        self.file_log.log(msg)

        self.model.to(self.device)
        self.model.eval()

        metrics = self._eval_with_metrics(dataloader, display_batches)
            
        msg = f"""Testing
                Pixel AUROC: {metrics["pixel_auroc"]}
                Pixel AP: {metrics["pixel_ap"]}
                IoU: {metrics["iou"]}
                F1: {metrics["f1"]}
                Pixel Accuracy: {metrics["pixel_accuracy"]}
                """
        
        logging.info(msg)
        self.file_log.log(msg)
        
        return metrics
    

    def _initialize_decoder_weights(self):
        """
        Applies Kaiming (He) initialization to all layers within the decoder.
        """
        def _apply_init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize only decoder layers -- encoder is pretrained
        self.model.decoder.apply(_apply_init)


    def _eval_with_metrics(self,
                           dataloader: torch.utils.data.DataLoader,
                           show_batches: list[int] =None) -> dict[str, float]:
        """
        Evaluate the model on the given dataloader and return metrics.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader containing the testing data.
            show_batches (list[int], optional): A list of batch indices that will be displayed. Defaults to None.
            
        Returns:
            dict[str, float]: A dictionary containing the metrics of the model.
        """
        self.model.eval()

        with torch.inference_mode():
            metrics = {"pixel_auroc": 0.0,
                    "pixel_ap": 0.0,
                    "iou": 0.0,
                    "f1": 0.0,
                    "pixel_accuracy": 0.0
                    }

            for i, batch in enumerate(dataloader, 1):
                images, masks = batch
                images = images.to(self.device)
                masks = masks.numpy()

                # Supervision on = 4 outputs / Supervision off = 1 output
                outputs = self.model(images)
                if self.supervision:
                    outputs = torch.stack(outputs)
                    outputs = torch.mean(outputs, dim=0)

                # Logits converted to probabilities
                output = torch.sigmoid(outputs)
                output = output.cpu().numpy()

                # Metrics per sample
                for j in range(output.shape[0]):
                    sample_metrics = evaluate_metrics(output[j], masks[j], self.threshold)
                    # Invalid sample (empty mask)
                    if sample_metrics is None: continue

                    metrics["pixel_auroc"] += sample_metrics["pixel_auroc"]
                    metrics["pixel_ap"] += sample_metrics["pixel_ap"]
                    metrics["iou"] += sample_metrics["iou"]
                    metrics["f1"] += sample_metrics["f1"]
                    metrics["pixel_accuracy"] += sample_metrics["pixel_accuracy"]

                ### Displaying
                if i in (show_batches or []):
                    # Display the batch
                    for j in range(dataloader.batch_size):
                        input = images[j].cpu().numpy().transpose(1, 2, 0)
                        mask = masks[j].transpose(1, 2, 0)
                        raw = output[j].transpose(1, 2, 0)
                        thresholded = threshold_pred(raw, self.threshold)

                        separate_visual([input, mask, raw, thresholded], ["Input image", "Ground truth mask", "Raw mask", "Thresholded mask"])

            # Average metrics
            metrics["pixel_auroc"] /= len(dataloader.dataset)
            metrics["pixel_ap"] /= len(dataloader.dataset)
            metrics["iou"] /= len(dataloader.dataset)
            metrics["f1"] /= len(dataloader.dataset)
            metrics["pixel_accuracy"] /= len(dataloader.dataset)

        return metrics
    

    def _freeze_backbone(self, freeze: bool):
        for param in self.model.encoder.parameters():
            param.requires_grad = not freeze

        if freeze:
            self.model.encoder.eval()
        else:
            self.model.encoder.train()


    def _export_metrics_plots(self, epochs: int, val_interval: int):
        """
        Generates and saves plots for training loss and validation metrics.

        Args:
            epochs (int): Total number of epochs trained.
            val_interval (int): The interval at which validation was performed.
        """
        path = os.path.join(self.logs_dir, f"{self.model_name}_plots")
        # Loss
        save_metric_plot(self.train_metrics["loss"], range(1, epochs+1), "Loss", f"{self.model_name}_loss", path)
        
        if val_interval and val_interval > 0 and val_interval < epochs:
            save_metric_plot(self.train_metrics["pixel_auroc"], range(1, epochs+1, val_interval), "Pixel AUROC", f"{self.model_name}_pixel_auroc", path)
            save_metric_plot(self.train_metrics["pixel_ap"], range(1, epochs+1, val_interval), "Pixel AP", f"{self.model_name}_pixel_ap", path)
            save_metric_plot(self.train_metrics["iou"], range(1, epochs+1, val_interval), "IoU", f"{self.model_name}_iou", path)
            save_metric_plot(self.train_metrics["f1"], range(1, epochs+1, val_interval), "F1", f"{self.model_name}_f1", path)
            save_metric_plot(self.train_metrics["pixel_accuracy"], range(1, epochs+1, val_interval), "Pixel Accuracy", f"{self.model_name}_pixel_accuracy", path)

        logging.info(f"Metrics plots exported to {path}")


    def _save_checkpoint(self, epoch: int):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "f1": float(self.best_f1)
        }

        torch.save(checkpoint, self.best_checkpoint_name)