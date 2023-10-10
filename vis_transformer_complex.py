import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ComplexScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        
        # Compute the complex dot product
        scores_real = torch.matmul(torch.real(Q), torch.real(K).transpose(-2, -1)) + \
                      torch.matmul(torch.imag(Q), torch.imag(K).transpose(-2, -1))
        scores_imag = torch.matmul(torch.real(Q), torch.imag(K).transpose(-2, -1)) - \
                      torch.matmul(torch.imag(Q), torch.real(K).transpose(-2, -1))
        
        # Combine to get complex scores
        scores = torch.complex(scores_real, scores_imag) / (d_k ** 0.5)
        
        # For the purpose of attention, we might just consider the magnitude of the scores
        scores_magnitude = torch.abs(scores)
        
        if mask is not None:
            scores_magnitude = scores_magnitude.masked_fill(mask == 0, -1e9)
        
        p_attn = F.softmax(scores_magnitude, dim=-1)
        p_attn = self.dropout(p_attn)
        
        # Using the attention probabilities, we compute the final complex values
        output_real = torch.matmul(p_attn, torch.real(V))
        output_imag = torch.matmul(p_attn, torch.imag(V))
        
        return torch.complex(output_real, output_imag), p_attn


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.real_layer = nn.Linear(in_features, out_features, bias=bias)
        self.imag_layer = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        real_part, imag_part = torch.real(x), torch.imag(x)
        out_real = self.real_layer(real_part) - self.imag_layer(imag_part)
        out_imag = self.real_layer(imag_part) + self.imag_layer(real_part)
        return torch.complex(out_real, out_imag)

class ComplexTransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        # Multi-head attention and normalization
        self.attn = ComplexMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = ComplexLayerNorm(d_model)
        
        # Feed-forward neural network and normalization
        self.ff = nn.Sequential(
            ComplexLinear(d_model, ff_dim),
            nn.ReLU(),  # ReLU can be applied to complex values directly
            ComplexLinear(ff_dim, d_model)
        )
        self.norm2 = ComplexLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward neural network
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
