import torchvision.models as models
import torch.nn as nn


def get_model(num_classes, model_name='resnet152'):
    """
    Create model with specified architecture
    
    Args:
        num_classes: Number of output classes
        model_name: Model architecture to use ('resnet50', 'resnet152', etc.)
    """
    if model_name == 'resnet50':
        model = models.resnet50(weights="IMAGENET1K_V1")
    elif model_name == 'resnet152':
        model = models.resnet152(weights="IMAGENET1K_V1")
    elif model_name == 'resnet101':
        model = models.resnet101(weights="IMAGENET1K_V1")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    in_features = model.fc.in_features

    # Replace the final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
        # nn.Sigmoid()  # multi-label classification - applied in loss function
    )
    return model