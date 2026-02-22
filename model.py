import timm
import torch.nn as nn

def get_model(num_classes):

    model = timm.create_model(
        'convit_tiny',
        pretrained=True,
        num_classes=0
    )

    model.head = nn.Sequential(
        nn.Linear(model.num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model