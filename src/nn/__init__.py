import torch
import pytorch_lightning as pl


class LSTMNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Feature Layers
        self.features = torch.nn.Sequential(
            torch.nn.LSTM(1, 32, 1),

            torch.nn.BatchNorm1d(380, eps=0.001, momentum=0.99),
            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(p=0.15),
        )

        # Classification Layers
        # Core Layer - Dense(number of nodes, activation function)
        # Noise Layer (Used to avoid overfiting) - Dropout(dropout percentage)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(380, 380),

            torch.nn.BatchNorm1d(380, eps=0.001, momentum=0.99),
            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(p=0.75),

            torch.nn.Linear(380, 4)
        )

        # Loss, train acc and valid acc
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, X, **kwargs):
        X = self.features(X)
        X = torch.flatten(X, start_dim=1)
        X = self.classifier(X)

        return X

    def configure_optimizers(self):
        return torch.optim.Adamax(self.parameters(), lr=1e-3, eps=1e-07)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = pl.metrics.functional.accuracy(logits, y)

        self.log('train_loss', loss)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = pl.metrics.functional.accuracy(logits, y)

        self.log('val_loss', loss)
        self.log('valid_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
