"""
test_model.py

このスクリプトでは、`train_model.py`でトレーニングされたモデルを使用して、MNISTデータセット上でモデルのパフォーマンスをテストします。
ロードされたモデルは、MNISTデータセットのテスト画像に対する予測を行い、その結果を表示します。

主な機能:
- トレーニング済みのCNNモデルのパラメータを読み込みます。
- MNISTデータセットのテスト画像に対して予測を行います。
- 予測結果と実際のラベルを比較し、結果を表示します。

使用方法:
スクリプトを実行して、トレーニング済みモデルのパフォーマンスをテストします。
    python test_model.py
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets

from train_model import Net


# 画像変換の設定
transform = transforms.Compose([transforms.ToTensor()])

# ネットワークモデルのロード
net = Net().cpu().eval()

# 重みの読み込み
model_weights = torch.load("mnist.pt", map_location=torch.device("cpu"))

# 読み込んだ重みをネットワークモデルに設定
net.load_state_dict(model_weights)

# 画像のロード
test = datasets.MNIST("./", train=False, download=False, transform=transform)
x, t = test[0]

# 予測値の算出
y = net(x.unsqueeze(0))

# 確率に変換
y = F.softmax(y, dim=1)
y = torch.argmax(y)

print(f"正解：{t}", f"予測：{y}")
