import torch


class EmbeddedModel:
    @staticmethod
    def cnn():
        return torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, 64, stride=2, padding=32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(8, stride=1, padding=0),

            torch.nn.Conv1d(16, 32, 32, stride=2, padding=16),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(8, stride=1, padding=0),

            torch.nn.Conv1d(32, 64, 16, stride=2, padding=8),
            torch.nn.ReLU(),

            torch.nn.Conv1d(64, 200, 1, stride=1, padding=1),
            torch.nn.ReLU(),

            torch.nn.MaxPool1d(8, stride=1, padding=0),

            torch.nn.Flatten(),

            torch.nn.Linear(3200, 200)
        )
