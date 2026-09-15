import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """
    Temporal self-attention module.

    It gives different importance to different
    frames in the video sequence.

    For example:

        Frame 1 → low importance
        Frame 2 → medium importance
        Frame 3 → high importance  ← decisive action
        Frame 4 → medium importance
    """

    def __init__(self, hidden_size):
        super().__init__()

        # Bi-LSTM produces hidden_size * 2 features
        # because it works in both forward and backward
        # directions.
        self.score = nn.Linear(
            hidden_size * 2,
            1
        )

    def forward(self, x):

        # Calculate an attention score for every frame
        scores = self.score(x)

        # Convert scores into probabilities.
        # The weights across all frames sum to 1.
        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        # Create a weighted combination of the
        # temporal features.
        context = torch.sum(
            x * attention_weights,
            dim=1
        )

        return context, attention_weights


class ActionRecognitionModel(nn.Module):
    """
    Complete temporal action-recognition model.

    Architecture:

        ResNet-50 features
              ↓
          2-layer Bi-LSTM
              ↓
        Self-Attention
              ↓
           Classifier
              ↓
        Action prediction
    """

    def __init__(
        self,
        input_size=2048,
        hidden_size=256,
        num_layers=2,
        num_classes=6,
        dropout=0.3
    ):
        super().__init__()

        # Bi-LSTM processes the sequence in both
        # forward and backward directions.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Attention decides which frames are
        # more important for classification.
        self.attention = TemporalAttention(
            hidden_size
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                128
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        # x:
        # [batch_size, sequence_length, 2048]
        #
        # Example:
        # [1, 32, 2048]
        #
        # 1   = video
        # 32  = sampled frames
        # 2048 = ResNet features
        lstm_output, _ = self.lstm(x)

        # Select important temporal information
        context, attention_weights = (
            self.attention(lstm_output)
        )

        # Predict the action class
        logits = self.classifier(context)

        return (
            logits,
            attention_weights
        )