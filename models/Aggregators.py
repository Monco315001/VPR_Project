import torch
from torch import nn
from torch.nn import functional as F
from models.Backbone import get_backbone

class Flatten(nn.Module):  # override of the flatten class
    """
    Custom module to flatten a tensor to 2D.
    """
    def __init__(self):
        super().__init__()  
        
    def forward(self, x):
        # ensures the tensor has the third and fourth dimensions equal to 1
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"  
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    """
    Custom module to perform L2 normalization along a specified dimension.
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim  
    
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)  # perform L2 normalization
    

class Avg_ResNet(nn.Module):
    """
    Defines the Avg_ResNet model which utilizes a pre-trained ResNet18 as its backbone 
    and incorporates an aggregation layer that performs average pooling.
    """
    def __init__(self):  
        super(Avg_ResNet, self).__init__()
        self.backbone = get_backbone()
        self.aggregation = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(256, 256),  
                L2Norm()  
            ) 
    def forward(self, x):
        x = self.backbone(x)  
        x = self.aggregation(x)  
        return x


# Gem pooling layer to obtain the final embedding
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        """
        Initializes the GeM pooling layer.
        
        Args:
            p (float): The power parameter for GeM pooling.
            eps (float): A small epsilon value to avoid division by zero.
        """
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        """
        Forward pass for the GeM pooling layer.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Pooled output tensor.
        """
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        """
        Performs the GeM pooling operation.
        
        Args:
            x (torch.Tensor): Input tensor.
            p (float): The power parameter for GeM pooling.
            eps (float): A small epsilon value to avoid division by zero.
            
        Returns:
            torch.Tensor: GeM pooled tensor.
        """
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

gem_pool = GeM()  # Instantiate the GeM pooling layer


# Network with truncated ResNet-18 followed by gem pooling
class GeM_ResNet(nn.Module):
    """
    Defines the GeM_ResNet model which uses a pre-trained ResNet18 as its backbone
    and incorporates an aggregation layer that performs gem pooling.
    """
    def __init__(self):
        super(GeM_ResNet, self).__init__()
        # Load the pretrained ResNet-18 model
        self.backbone = get_backbone()
        self.aggregation = nn.Sequential(                   
                # L2Norm(),                                   
                gem_pool,
                Flatten(),
                nn.Linear(256, 256),     
                # L2Norm()                                    
            )                                               

    def forward(self, x):
        x = self.backbone(x)                                
        x = self.aggregation(x)                             
        return x  
    