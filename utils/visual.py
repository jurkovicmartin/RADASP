import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


def separate_visual(images: list, titles: list):
    """
    Visualize multiple images in separate subplots.

    Args:
        images (list): List of images to visualize.
        titles (list): List of titles for each image.
    """
    num_images = len(images)

    if num_images != len(titles):
        raise ValueError("Number of images and titles must be the same.")
    
    fig, axes = plt.subplots((num_images + 1) // 2, 2, figsize=(8, 4))

    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def overlap_visual(img: np.ndarray, mask: np.ndarray, title: str = "Image"):
    """
    Visualize the image with the mask overlapping on top of it.

    Args:
        img (np.ndarray): Input image.
        mask (np.ndarray): Input mask.
        title (str, optional): Title for the plot. Defaults to "Image".
    """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.imshow(mask, cmap="Reds", alpha=0.5)
    ax.axis("off")
    plt.title(title)
    plt.show()


def mask_comparison_visual(mask_a: np.ndarray, mask_b: np.ndarray, title_a="Mask A", title_b="Mask B"):
    """
    Displays a comparison of two masks by creating a difference map.

    Args:
        mask_a (np.ndarray): The first mask to compare.
        mask_b (np.ndarray): The second mask to compare.
        title_a (str, optional): The title of mask A. Defaults to "Mask A".
        title_b (str, optional): The title of mask B. Defaults to "Mask B".
    """
    # Boolean for logical operations
    mask_a_bool = mask_a.astype(bool).squeeze(-1)
    mask_b_bool = mask_b.astype(bool).squeeze(-1)

    # Create difference map
    diff_img = np.full((*mask_a_bool.shape, 3), 220, dtype=np.uint8)
    INTERSECTION_COLOR = [0, 0, 139]
    A_COLOR = [255, 0, 255]
    B_COLOR = [0, 255, 0]  
    diff_img[mask_a_bool & mask_b_bool] =  INTERSECTION_COLOR
    diff_img[mask_a_bool & ~mask_b_bool] = A_COLOR
    diff_img[~mask_a_bool & mask_b_bool] = B_COLOR
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Mask A
    axes[0].imshow(mask_a, cmap="gray")
    axes[0].set_title(title_a)
    axes[0].axis("off")
    
    # Mask B
    axes[1].imshow(mask_b, cmap="gray")
    axes[1].set_title(title_b)
    axes[1].axis("off")
    
    # Difference
    axes[2].imshow(diff_img)
    axes[2].set_title("Difference Map")
    axes[2].axis("off")
    
    # Legend
    patch_a = mpatches.Patch(color=[c/255 for c in A_COLOR], label=f"Only in {title_a}")
    patch_b = mpatches.Patch(color=[c/255 for c in B_COLOR], label=f"Only in {title_b}")
    patch_inter = mpatches.Patch(color=[c/255 for c in INTERSECTION_COLOR], label="Intersection")
    
    axes[2].legend(handles=[patch_a, patch_b, patch_inter], 
                  loc="lower center", bbox_to_anchor=(0.5, -0.2), 
                  ncol=3, frameon=False)

    plt.tight_layout()
    plt.show()


def save_metric_plot(values: list[float], epochs: list[int], metric: str, filename: str ="plot", save_path: str ="/plots"):
    """
    Save the metric plot to a file -- Metric development over epochs.

    Args:
        values (list[float]): List of metric values.
        epochs (list[int]): List of epochs.
        metric (str): Name of the metric.
        filename (str, optional): Name of the file. Defaults to "plot".
        save_path (str, optional): Path to save the plot. Defaults to "/plots".
    """
    if len(values) != len(epochs):
        raise ValueError("Values and epochs must have the same length.")
    
    os.makedirs(save_path, exist_ok=True)

    plt.plot(epochs, values)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.savefig(os.path.join(save_path, f"{filename}.png"))
    plt.close()


