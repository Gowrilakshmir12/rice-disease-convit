import timm
import torch
import torch.nn as nn

num_classes = 5   

model = timm.create_model('convit_tiny', pretrained=True)

model.head = nn.Linear(model.head.in_features, num_classes)

print(model.head)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(next(model.parameters()).device)


