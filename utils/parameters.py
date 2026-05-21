import torch.nn as nn

def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to analyze.

    Returns:
        int: Total number of parameters (trainable and non-trainable).
    """
    total_trainable_params = 0
    total_non_trainable_params = 0

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            total_trainable_params += parameter.numel()
        else:
            total_non_trainable_params += parameter.numel()
            
    total_params = total_trainable_params + total_non_trainable_params
    
    return total_params



if __name__ == "__main__":
    from architecture.radasp import RADASP


    params1 = count_parameters(RADASP(1, 1, True))
    encoder_params1 = count_parameters(RADASP(1, 1, True).encoder)
    decoder_params1 = count_parameters(RADASP(1, 1, True).decoder)
    print("RADASP\n" + 20 * "-")
    print(f"Total parameters: {params1:,}")
    print(f"Encoder parameters: {encoder_params1:,}")
    print(f"Decoder parameters: {decoder_params1:,}")




       
