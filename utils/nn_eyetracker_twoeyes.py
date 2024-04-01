"""
EyeTracker for predicting the final gaze points
This neural network takes only the left eye and the right eye
and need therefore less computing time
"""
from torch import nn
import torch

class EyeTracker(nn.Module):
    """EyeTracker torch model."""

    def __init__(self):
        """Initialize model.
        """
        super().__init__()

        # Left eye
        self.eye = nn.Sequential(
            # input size 3 x 50 x 100
            nn.Conv2d(3, 12, 5, bias=False),
            # size: 12 x 46 x 96
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # size: 12 x 23 x 48
            nn.Dropout2d(0.5),

            nn.Conv2d(12, 24, 4, bias=False),
            # size: 24 x 20 x 45
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # size: 24 x 10 x 22
            nn.Dropout2d(0.5),

            nn.Conv2d(24, 64, 3, bias=False),
            # size: 64 x 8 x 20
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # size: 64 x 4 x 10
            nn.Dropout2d(0.5),

            nn.Flatten()
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(2 * 64 * 4 * 10, 512),
            # nn.Linear(128 * 5 * 45 + 2 * 64 * 4 * 10, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 2)
        )

    def forward(self, left_eye, right_eye):
        result_le = self.eye(left_eye)
        result_re = self.eye(right_eye)

        result = torch.cat((result_le,result_re), dim = 1)
        output = self.fc(result)

        return output

