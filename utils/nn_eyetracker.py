"""
EyeTracker for predicting the final gaze points
This neural network takes the left eye, the right eye and the eye region
and need therefore a lot of computing time
"""
from torch import nn
import torch

class EyeTracker(nn.Module):
    """EyeTracker torch model."""

    def __init__(self):
        """Initialize model.
        """
        super().__init__()

        # one eye
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

        self.eyeregion = nn.Sequential(
            # input size 3 x 80 x 400
            nn.Conv2d(3, 24, 9, bias=False),
            # size: 24 x 72 x 392
            nn.BatchNorm2d(24, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # size: 24 x 36 x 196
            nn.Dropout2d(0.5),

            nn.Conv2d(24, 64, 7, bias=False),
            # size: 64 x 30 x 1903
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # size: 64 x 15 x 95
            nn.Dropout2d(0.5),

            nn.Conv2d(64, 128, 6, bias=False),
            # size: 128 x 10 x 90
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # size: 128 x 5 x 45
            nn.Dropout2d(0.5),

            nn.Conv2d(128, 256, 4, bias=False),
            # size: 256 x 2 x 42
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # size: 256 x 1 x 21
            nn.Dropout2d(0.5),  # Dropout layer

            nn.Flatten()
        )

        # self.eyeregion = nn.Sequential(
        #     # input size 3 x 80 x 400
        #     nn.Conv2d(3, 9, 9, bias=False),
        #     # size: 9 x 72 x 392
        #     nn.BatchNorm2d(9, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(2),
        #     # size: 9 x 36 x 196
        #     nn.Dropout2d(0.5),
        #
        #     nn.Conv2d(9, 18, 7, bias=False),
        #     # size: 18 x 30 x 1903
        #     nn.BatchNorm2d(18, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(2),
        #     # size: 18 x 15 x 95
        #     nn.Dropout2d(0.5),
        #
        #     nn.Conv2d(18, 36, 6, bias=False),
        #     # size: 36 x 10 x 90
        #     nn.BatchNorm2d(36, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(2),
        #     # size: 36 x 5 x 45
        #     nn.Dropout2d(0.5),
        #
        #     nn.Conv2d(36, 64, 4, bias=False),
        #     # size: 64 x 2 x 42
        #     nn.BatchNorm2d(64, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(2),
        #     # size: 64 x 1 x 21
        #     nn.Dropout2d(0.5),  # Dropout layer
        #
        #     nn.Flatten()
        # )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(256 * 21 * 1 + 2 * 64 * 4 * 10, 512),
            # nn.Linear(64 * 21 * 1 + 2 * 64 * 4 * 10, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 2)
        )

    def forward(self, eye_region, left_eye, right_eye):
        result_er = self.eyeregion(eye_region)
        result_le = self.eye(left_eye)
        result_re = self.eye(right_eye)

        result = torch.cat((result_er,result_le,result_re), dim = 1)
        output = self.fc(result)

        return output

