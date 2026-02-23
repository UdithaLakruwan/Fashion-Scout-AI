import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionModel(nn.Module):
  def __init__(self):
    super().__init__()
    #Look for 16 patterns in the 1-channel image
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  #Grayscaleimage : channel = 1, Colourimage: channel = 3

    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    #Shrink the image by half (28x28 becomes 14x14)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # Dropout to prevent overfitting
    self.dropout = nn.Dropout(0.25)

    #Flatten the 3D data into a 1D line
    self.flat = nn.Flatten()

    # Calculation: 32 filters * 14px * 14px = 6272
    self.fc1 = nn.Linear(6272, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = self.dropout(x)
    x = self.flat(x)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x