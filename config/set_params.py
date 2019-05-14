import torch

class params:
    """Modify this class to set parameters."""
    def __init__(self):
        self.params = {
            "root": "data/pml-training.csv",
            "test": "data/pml-testing.csv",
            "resume": None,
            "input_dim": 1,
            "num_classes": 5,
            "workers": 4,
            "batch_size": 32,
            "epochs": 300,
            "lr": 0.003,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "split": 0.2,
            "use_cuda": torch.cuda.is_available(),
        }