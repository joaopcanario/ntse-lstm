import os
import random
import torch

import numpy as np
import pytorch_lightning as pl
import sklearn.metrics as metrics

from loguru import logger
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.lstm import LSTMFrame, LSTMCell
from src import viz


logger.add("log_{time}.log", format="{time} {level} {message}", level="INFO")
kwargs = {'gpus': 1, 'accelerator': 'dp', 'max_epochs': 10, 'deterministic': True}


class LSTMNet(pl.LightningModule):
    def __init__(self, lstm_layer=None):
        super().__init__()
        # Feature Layers
        self.lstm = lstm_layer

        self.relu = torch.nn.ReLU()

        self.dp = torch.nn.Dropout(p=0.15)
        # Classification Layers
        # Core Layer - Dense(number of nodes, activation function)
        # Noise Layer (Used to avoid overfiting) - Dropout(dropout percentage)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(300 * 200, 380),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.75),
            torch.nn.Linear(380, 4)
        )

    def forward(self, X, **kwargs):
        h0 = torch.randn(1, X.size(0), 200).requires_grad_()
        c0 = torch.randn(1, X.size(0), 200).requires_grad_()

        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        X, _ = self.lstm(X, (h0, c0))
        X = self.relu(X)
        X = self.dp(X)

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
        f1 = pl.metrics.functional.f1(logits, y, 4)

        self.log('train_loss', loss)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = pl.metrics.functional.accuracy(logits, y)
        f1 = pl.metrics.functional.f1(logits, y, 4)

        self.log('val_loss', loss)
        self.log('valid_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid_f1', f1, on_step=False, on_epoch=True, prog_bar=True)


def eval_net(net, X_test):
    net.eval()

    if torch.cuda.is_available():
        net.cuda()

        preds = np.concatenate([net(t).cpu().detach().numpy()
                                for t in torch.chunk(X_test.cuda(), 77, dim=0)],
                               axis=0)
    else:
        preds = net(X_test).cpu().detach().numpy()

    y_pred = np.argmax(preds, axis=1)

    return y_pred


def train_eval(layer, X, y, X_test, y_test, name):
    accs, fscores = [], []
    bigger = 0

    for i in range(1, 11):
        logger.info(f"Fold: {i}")

        trainloader, valloader = load_fold(i, X, y)

        net = LSTMNet(layer)
        pl.Trainer(**kwargs).fit(net, trainloader, valloader)

        y_pred = eval_net(net, X_test)

        acc = metrics.accuracy_score(y_test.numpy(), y_pred)
        fscore = metrics.f1_score(y_test.numpy(), y_pred, average=None).mean()

        logger.info(f"Partial result: {acc * 100:.2f}%, {fscore * 100:.2f}%")

        accs.append(acc)
        fscores.append(fscore)

        if fscore > bigger:
            logger.info(f"Saving best model - fold: {i}")
            bigger = fscore

            PATH = Path(f"data/{name}.pt")
            PATH.unlink(missing_ok=True)

            torch.save(net.state_dict(), PATH)

    logger.info("ACC - F1-Score")
    logger.info(f"\n\n{viz.metrics(accs, fscores)}")


def noise_test_eval(net):
    noises = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    y = torch.load('data/ytest.pt')

    for noise in noises:
        logger.info(f"== Testing - Noise Level ({int(noise * 100)}%) == ")

        X_noisy = torch.load(f'data/Xtest_{noise}.pt')
        y_pred = eval_net(net, X_noisy)

        cm = metrics.confusion_matrix(y.numpy(), y_pred)
        logger.info(f"\n\n{viz.classification_report([cm])}\n")


def seed_all(seed):
    logger.info(f"Lock seed: {seed}")

    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


def kfold_experiment():
    logger.info("== Start train/eval experiment ==")
    logger.info("Loading training set")

    with open('data/x_y_trainset.npy', 'rb') as f:
        X = np.load(f)
        y = np.load(f)

    logger.info("Loading test set")

    with open('data/x_y_testset.npy', 'rb') as f:
        X_test = np.load(f)
        y_test = np.load(f)

    logger.info("Prepare data")

    X_train = X.reshape(-1, 300, 20)

    X_test = torch.tensor(X_test.reshape(-1, 300, 20)).float()
    y_test = torch.tensor(y_test).argmax(dim=-1)

    logger.info("Start Original k-fold eval")

    original = torch.nn.LSTM(20, 200, batch_first=True)
    train_eval(original, X_train, y, X_test, y_test, "origin")

    logger.info("Start Modified k-fold eval")

    modified = LSTMFrame([[LSTMCell(20, 200)]], dropout=0.0,
                         bidirectional=False, batch_first=True)
    train_eval(modified, X_train, y, X_test, y_test, "modified")

    logger.info("End of train/eval experiment")


def load_fold(fold, X, y):
    with open(f'data/train_fold_{fold}.npy', 'rb') as t, \
            open(f'data/val_fold_{fold}.npy', 'rb') as v:

        train = np.load(t)
        val = np.load(v)

    train_tensor = torch.tensor(X[train]).float()
    val_tensor = torch.tensor(X[val]).float()

    train_data = TensorDataset(
        train_tensor, torch.tensor(y[train]).argmax(dim=-1))
    val_data = TensorDataset(
        val_tensor, torch.tensor(y[val]).argmax(dim=-1))

    trainloader = DataLoader(train_data, batch_size=16, drop_last=True,
                             num_workers=os.cpu_count(), shuffle=True)
    valloader = DataLoader(val_data, batch_size=16, drop_last=True,
                           num_workers=os.cpu_count(), shuffle=False)

    return trainloader, valloader


def noise_experiment():
    logger.info("== Start Noise experiment ==")
    logger.info("Loading training set")

    logger.info("Training original LSTM model")

    original_net = LSTMNet(torch.nn.LSTM(20, 200, batch_first=True))
    original_net.load_state_dict(torch.load("data/origin.pt"))

    logger.info("Eval original model through multiple noise levels")

    noise_test_eval(original_net)

    logger.info("Training Modified LSTM model")

    modified_net = LSTMNet(LSTMFrame([[LSTMCell(20, 200)]], dropout=0.0,
                                     bidirectional=False, batch_first=True))
    modified_net.load_state_dict(torch.load("data/modified.pt"))

    logger.info("Eval modified model through multiple noise levels")

    noise_test_eval(modified_net)

    logger.info("End of noise experiment")


if __name__ == "__main__":
    logger.info("Starting experiments")

    seed_all(42)

    kfold_experiment()
    noise_experiment()
