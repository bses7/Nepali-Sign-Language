import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
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
    def __init__(self, vocab_size, feature_dim=231, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        """
        NSL Transformer updated for 231-dim features:
        99 (Pose) + 63 (LH) + 63 (RH) + 3 (LH Wrist Meta) + 3 (RH Wrist Meta)
        """
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.text_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Maps 231 features into the transformer's d_model space
        self.motion_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048, 
            dropout=dropout,     
            batch_first=True,
            activation='gelu'
        )
        
        # Maps transformer output back to 231 features
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, feature_dim)
        )

    def forward(self, src_tokens, tgt_motion):
        """
        src_tokens: [Batch, Text_Len] (e.g., [SOS, "क", <TRANS>, "ख", EOS])
        tgt_motion: [Batch, Frame_Len, 231]
        """
        src = self.embedding(src_tokens) * math.sqrt(self.d_model)
        src = self.text_encoder(src)
        src = self.pos_encoder(src)
        
        tgt = self.motion_projection(tgt_motion)
        tgt = self.pos_encoder(tgt)
        
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_key_padding_mask = (src_tokens == 0) 
        
        output = self.transformer(
            src, 
            tgt, 
            tgt_mask=tgt_mask, 
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=None, 
            memory_key_padding_mask=src_key_padding_mask
        )
        
        return self.output_layer(output)

    def generate_step(self, src_tokens, current_motion):
        """Helper for inference to generate one frame at a time."""
        with torch.no_grad():
            output = self.forward(src_tokens, current_motion)
            return output[:, -1:, :]