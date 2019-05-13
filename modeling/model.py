import torch
import torch.nn as nn

class HARmodel(nn.Module):
    """
    Network to extract word, phrase, and question level features
    """

    def __init__(self, input_size, num_classes):
        super().__init__()

        self._cuda = torch.cuda.is_available()

        self.features = nn.Sequential(
            nn.Conv1d(input_size, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            )

        self.classifier = nn.Sequential(
        	nn.Dropout(),
        	nn.Linear(120, 128),
        	nn.ReLU(),
        	nn.Dropout(),
        	nn.Linear(128, num_classes),
        	)

    def forward(self, x):
    	x = self.features(x)
    	x = x.view(x.size(), 120)
    	out = self.classifier(x)

    	return out