"""
train_model.py

このスクリプトでは、畳み込みニューラルネットワーク(CNN)モデルを構築、トレーニングし、そのパラメータを保存する方法を示します。
トレーニング後、モデルのパラメータを、FastAPIアプリケーションのために保存します。

主な機能:
- MNIST数字分類のためのCNNモデルを定義します。
- PyTorch Lightningフレームワークを使用してモデルをトレーニングします。
- トレーニングされたモデルのパラメータを保存します。

使用方法:
スクリプトを実行してモデルをトレーニングし、そのパラメータを保存します。
    python train_model.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from torchvision import datasets

from torchmetrics.functional import accuracy
from pytorch_lightning.loggers import CSVLogger


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(588, 10)

    def forward(self, x):
        h = self.conv(x)
        h = F.relu(h)
        h = self.bn(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = h.view(-1, 588)
        h = self.fc(h)
        return h

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            accuracy(y.softmax(dim=-1), t, task="multiclass", num_classes=10, top_k=1),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "val_acc",
            accuracy(y.softmax(dim=-1), t, task="multiclass", num_classes=10, top_k=1),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "test_acc",
            accuracy(y.softmax(dim=-1), t, task="multiclass", num_classes=10, top_k=1),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer


def main():
    # データセットの変換を定義
    transform = transforms.Compose([transforms.ToTensor()])

    # データセットの取得
    train_val = datasets.MNIST("./", train=True, download=True, transform=transform)

    # train と val に分割
    n_train = 50000
    n_val = 10000

    torch.manual_seed(0)
    train, val = torch.utils.data.random_split(train_val, [n_train, n_val])

    # バッチサイズの定義
    batch_size = 1024

    # Data Loader を定義
    train_loader = torch.utils.data.DataLoader(
        train, batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(val, batch_size)

    # 学習の実行
    pl.seed_everything(0)
    net = Net()
    logger = CSVLogger(save_dir="logs", name="my_exp")
    trainer = pl.Trainer(
        max_epochs=10, accelerator="cpu", deterministic=False, logger=logger
    )
    trainer.fit(net, train_loader, val_loader)

    # 学習済みモデルの保存
    torch.save(net.state_dict(), "mnist.pt")


if __name__ == "__main__":
    main()
