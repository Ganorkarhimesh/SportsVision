import torch
import torch.nn as nn


class SportsVisionModel(nn.Module):
    """
    SportsVision temporal action recognition model.

    Input:
        [batch_size, 32, 2048]

    Output:
        [batch_size, 11]

    Architecture:
        ResNet-50 features
            ↓
        Bi-LSTM
            ↓
        Self-Attention
            ↓
        Attention Pooling
            ↓
        Fully Connected
            ↓
        11 Classes
    """

    def __init__(
        self,
        input_size=2048,
        hidden_size=256,
        num_layers=2,
        num_heads=8,
        num_classes=11,
        dropout=0.3
    ):
        super().__init__()

        # ------------------------------------------
        # Bi-LSTM
        # ------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Bi-LSTM output size:
        # 256 forward + 256 backward = 512

        lstm_output_size = hidden_size * 2


        # ------------------------------------------
        # Self-Attention
        # ------------------------------------------

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )


        # ------------------------------------------
        # Layer Normalization
        # ------------------------------------------

        self.layer_norm = nn.LayerNorm(lstm_output_size)


        # ------------------------------------------
        # Classifier
        # ------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )


    def forward(self, x):

        # x shape:
        # [batch_size, 32, 2048]

        # ------------------------------------------
        # Bi-LSTM
        # ------------------------------------------

        lstm_output, _ = self.lstm(x)

        # Shape:
        # [batch_size, 32, 512]


        # ------------------------------------------
        # Self-Attention
        # ------------------------------------------

        attention_output, _ = self.attention(
            lstm_output,
            lstm_output,
            lstm_output
        )

        # Shape:
        # [batch_size, 32, 512]


        # ------------------------------------------
        # Residual Connection + LayerNorm
        # ------------------------------------------

        attention_output = self.layer_norm(
            attention_output + lstm_output
        )


        # ------------------------------------------
        # Temporal Pooling
        # ------------------------------------------

        # Average all 32 frames

        pooled_output = attention_output.mean(dim=1)

        # Shape:
        # [batch_size, 512]


        # ------------------------------------------
        # Classification
        # ------------------------------------------

        output = self.classifier(pooled_output)

        # Shape:
        # [batch_size, 11]

        return output