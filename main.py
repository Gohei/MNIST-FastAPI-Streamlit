"""
main.py

このスクリプトでは、FastAPIを使用して、トレーニング済みのMNISTモデルを提供するAPIサーバーを構築します。
クライアントは、画像データをPOSTリクエストとしてサーバーに送信し、モデルによる数字の予測結果を受け取ります。

主な機能:
- FastAPIを使用したAPIサーバーの構築。
- トレーニング済みのCNNモデルを用いた画像の予測処理。
- 予測結果をJSON形式でクライアントに返却。

使用方法:
スクリプトを実行してAPIサーバーを起動し、
クライアントからのHTTP POSTリクエストで画像データを送信します。
    python main.py

"""

from fastapi import FastAPI, Request
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from train_model import Net


app = FastAPI()

# 画像変換の設定
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),  # MNISTの画像サイズに合わせる
        transforms.Grayscale(),  # グレースケールに変換
        transforms.ToTensor(),  # Tensor形式に変換
    ]
)

# ネットワークモデルのロード
net = Net().cpu().eval()

# 重みの読み込み
model_weights = torch.load("mnist.pt", map_location=torch.device("cpu"))

# 読み込んだ重みをネットワークモデルに設定
net.load_state_dict(model_weights)


@app.post("/prediction")
async def predict_image(request: Request):
    # リクエストから画像データをバイトとして取得
    image_bytes = await request.body()
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)

    # 画像をモデルに適した形に変換
    transformed_image = transform(image)

    # モデルを使用して予測
    prediction = net(transformed_image.unsqueeze(0))

    # 最も確率が高いクラスを選択
    prediction = F.softmax(prediction, dim=1)
    predicted_class = torch.argmax(prediction)

    # 予測結果を整数として返す
    return {"prediction": predicted_class.item()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
