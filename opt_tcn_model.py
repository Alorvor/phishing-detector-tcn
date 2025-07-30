import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        return F.relu(x + residual)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        scores = self.query(x).squeeze(-1)                 # [batch, seq_len]
        weights = torch.softmax(scores, dim=1)             # [batch, seq_len]
        context = torch.bmm(weights.unsqueeze(1), x)       # [batch, 1, hidden_dim]
        return context.squeeze(1), weights                 # [batch, hidden_dim], [batch, seq_len]


class TCNWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, tcn_channels=[64, 128, 256, 256],
                 kernel_size=3, dilations=[1, 2, 4, 8], dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # TCN stack
        tcn_blocks = []
        in_channels = embed_dim
        for out_channels, dilation in zip(tcn_channels, dilations):
            tcn_blocks.append(ResidualTCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
            in_channels = out_channels
        self.tcn = nn.Sequential(*tcn_blocks)

        self.attention = AttentionPooling(input_dim=tcn_channels[-1])

        self.fc = nn.Sequential(
            nn.Linear(tcn_channels[-1], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)      # [batch, embed_dim, seq_len]
        x = self.tcn(x)                             # [batch, channels, seq_len]
        x = x.permute(0, 2, 1)                      # [batch, seq_len, channels]
        context, att_weights = self.attention(x)    # [batch, channels], [batch, seq_len]
        out = self.fc(context)                      # [batch, 1]
        return out.squeeze(1), att_weights
