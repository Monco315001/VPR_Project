import torchvision
import torch

def get_backbone():
    """
    Loads a pre-trained ResNet18 model and modifies it by freezing certain layers and 
    removing the last two layers to create a custom backbone.

    Returns:
        torch.nn.Sequential: The modified backbone of ResNet18.
    """
    backbone = torchvision.models.resnet18(pretrained=True)  # loading the pre-trained model
    
    # Iterate through child modules in the backbone and freeze layers before "layer3"
    for name, child in backbone.named_children():
        if name == "layer3":
            break  # Freeze layers before conv_3 
        for params in child.parameters():
            params.requires_grad = False  # Freeze model parameters 

    layers = list(backbone.children())[:7]  # Remove the last two layers of the backbone (avg pooling and FC layer)
    backbone = torch.nn.Sequential(*layers)  # Create a backbone after manipulating the layers
    
    return backbone
