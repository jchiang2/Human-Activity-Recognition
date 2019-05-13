class params:
    def __init__(self):
        self.params = {
            "root": "data/dataset/pml-training.csv",
            "num_classes": 5,
            "workers": 4,
            "batch_size": 32,
            "epochs": 10000,
            "lr": 0.0003,
            "momentum": 0.9,
            "weight_decay": 1e-4,
        }