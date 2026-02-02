# Baseline CNN model for jersey number recognition
import torch
import torch.nn as nn
import torchvision.models as models


class BaselineCNN(nn.Module):
    """
    Baseline model: CNN backbone + frame averaging
    
    Processes each frame independently and averages predictions.
    """
    
    def __init__(
        self,
        num_classes: int = 100,  # 0-99 for jersey numbers
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim: int = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(feature_dim, feature_dim)  # type: ignore
            self.backbone.fc = nn.Identity()  # type: ignore
        
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim: int = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(feature_dim, feature_dim)  # type: ignore
            self.backbone.fc = nn.Identity()  # type: ignore
        
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = int(self.backbone.classifier[1].in_features)  # type: ignore
            self.backbone.classifier = nn.Sequential(nn.Identity())
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, aggregate: str = "mean"):
        """
        Args:
            x: Tensor of shape (batch_size, num_frames, C, H, W) or list of tensors
            aggregate: How to aggregate frame predictions ('mean', 'max', 'voting')
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # Handle variable-length sequences
        if isinstance(x, list):
            return self._forward_variable_length(x, aggregate)
        
        batch_size, num_frames, C, H, W = x.shape
        
        # Reshape to (batch_size * num_frames, C, H, W)
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Extract features
        features = self.backbone(x)  # (batch_size * num_frames, feature_dim)
        
        # Get per-frame predictions
        logits = self.classifier(features)  # (batch_size * num_frames, num_classes)
        
        # Reshape back to (batch_size, num_frames, num_classes)
        logits = logits.view(batch_size, num_frames, self.num_classes)
        
        # Aggregate across frames
        if aggregate == "mean":
            logits = logits.mean(dim=1)
        elif aggregate == "max":
            logits = logits.max(dim=1)[0]
        elif aggregate == "voting":
            # Soft voting: average probabilities
            probs = torch.softmax(logits, dim=2)
            probs = probs.mean(dim=1)
            logits = torch.log(probs + 1e-10)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
        
        return logits
    
    def _forward_variable_length(self, x_list, aggregate):
        """Handle variable-length sequences (list of tensors)."""
        batch_outputs = []
        
        for x in x_list:
            # x has shape (num_frames, C, H, W)
            num_frames = x.shape[0]
            
            # Extract features
            features = self.backbone(x)  # (num_frames, feature_dim)
            
            # Get per-frame predictions
            logits = self.classifier(features)  # (num_frames, num_classes)
            
            # Aggregate
            if aggregate == "mean":
                logits = logits.mean(dim=0, keepdim=True)
            elif aggregate == "max":
                logits = logits.max(dim=0, keepdim=True)[0]
            elif aggregate == "voting":
                probs = torch.softmax(logits, dim=1)
                probs = probs.mean(dim=0, keepdim=True)
                logits = torch.log(probs + 1e-10)
            
            batch_outputs.append(logits)
        
        # Stack outputs
        return torch.cat(batch_outputs, dim=0)


if __name__ == "__main__":
    # Test the model
    model = BaselineCNN(num_classes=100, backbone="resnet50")
    
    # Test with fixed-length batch
    x = torch.randn(4, 10, 3, 224, 224)  # 4 samples, 10 frames each
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (4, 100)
    
    # Test with variable-length batch
    x_list = [
        torch.randn(10, 3, 224, 224),
        torch.randn(15, 3, 224, 224),
        torch.randn(8, 3, 224, 224),
    ]
    output = model(x_list)
    print(f"Variable-length output shape: {output.shape}")  # Should be (3, 100)
