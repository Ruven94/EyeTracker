"""
EyeTracker torch model for predicting the gaze point.
"""
from torch import nn


class ClosedEyeDetection(nn.Module):
    """EyeTracker torch model."""

    def __init__(self):
        """Initialize model.
        """
        super().__init__()

        #  Eye
        self.conv = nn.Sequential(
            # input size 1 x 100 x 100
            nn.Conv2d(1, 12, 7, bias=False),
            # size: 8 x 94 x 94
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # size: 8 x 47 x 47
            nn.Dropout2d(0.5),

            nn.Conv2d(12, 36, 5, bias=False),
            # size: 16 x 43 x 43
            nn.BatchNorm2d(36),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # size: 16 x 21 x 21
            nn.Dropout2d(0.5),

            nn.Conv2d(36, 64, 3, bias=False),
            # size: 24 x 19 x 19
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # size: 24 x 9 x 9
            nn.Dropout2d(0.5),
            nn.Flatten()
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(64*9*9, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 2)
        )


    def forward(self, eye):
        result = self.conv(eye)
        result = self.fc(result)

        return result

