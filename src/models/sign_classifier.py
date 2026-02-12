import torch
import torch.nn as nn

class NSLClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim=225, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        self.input_proj = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True, dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1) 
        
        return self.classifier(x)