import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Helps the model understand the 'order' of frames in an animation.
    Without this, the model thinks Frame 1 and Frame 50 are the same.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class NSLTransformer(nn.Module):
    def __init__(self, vocab_size, feature_dim=225, d_model=256, nhead=8, num_layers=4, dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.motion_projection = nn.Linear(feature_dim, d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048, 
            dropout=dropout,     
            batch_first=True
        )
        self.output_layer = nn.Linear(d_model, feature_dim)

    def forward(self, src_tokens, tgt_motion):
        # 1. Encode Text (Source)
        src = self.embedding(src_tokens) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # 2. Project Motion (Target)
        tgt = self.motion_projection(tgt_motion) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # 3. Create Masks
        # Target mask (prevents looking ahead)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # KEY ADDITION: Padding masks
        # This tells the transformer to ignore the [PAD] tokens in the text
        src_key_padding_mask = (src_tokens == 0) # Assuming 0 is <PAD> id
        
        # 4. Run through Transformer
        # We explicitly separate the memory (text) from the sequence (motion)
        memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        return self.output_layer(output)