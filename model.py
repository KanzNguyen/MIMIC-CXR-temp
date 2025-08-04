import torchvision.models as models
import torch.nn as nn


def get_model(num_classes, model_name='resnet152'):
    """
    Create model optimized for medical feature extraction
    Fine-tune entire ResNet152 to learn medical-specific features
    
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
    
    # QUAN TRỌNG: Toàn bộ mô hình sẽ được fine-tune cho medical images
    # Không freeze bất kỳ layer nào để học medical-specific features
    
    in_features = model.fc.in_features  # ResNet152: 2048

    # Thay thế classifier head với architecture phù hợp cho medical imaging
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
        # Sigmoid sẽ được áp dụng trong BCEWithLogitsLoss
    )
    
    print(f"✅ Created {model_name} for medical feature extraction")
    print("🔥 Entire model will be fine-tuned for medical images")
    
    return model


def print_model_info(model):
    """Print model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🎯 Trainable parameters: {trainable_params:,}")
    print(f"🔥 All parameters are trainable for medical feature learning")