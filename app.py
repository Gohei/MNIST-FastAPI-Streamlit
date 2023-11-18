"""
app.py

このスクリプトは、Streamlitを利用して、ユーザーが手書きで画像を作成できる機能を提供します。
ユーザーは手書きで画像を作成し、そのバイトデータをFastAPIサーバーに送信して、
MNISTモデルによる数字の予測結果を取得します。

主な機能:
- Streamlitを使用したインタラクティブな手書き入力インターフェースの提供。
- 手書きされた画像データをFastAPIサーバーに送信し、予測を取得。
- 予測結果をStreamlitアプリケーション上で表示。

使用方法:
Streamlitアプリケーションとしてこのスクリプトを実行し、
ブラウザ上で手書き画像を作成し、予測を取得します。
    streamlit run app.py
"""

import streamlit as st
from streamlit_mnist_canvas import st_mnist_canvas

import requests
import pandas as pd

# アプリのタイトル
st.subheader("MNIST Handwritten Digit Recognition")

# 手書き入力インターフェースの表示
handwritten_image = st_mnist_canvas()

# 初期予測と確率の設定
prediction = ""
class_probabilities = [0.0] * 10

if handwritten_image.is_submitted:
    # FastAPIサーバーに画像を送信し、予測結果を取得
    response = requests.post(
        "http://127.0.0.1:8000/prediction", data=handwritten_image.raw_image_bytes
    )
    response_json = response.json()

    # 予測されたクラスと確率を安全に取得
    prediction = response_json.get("most_probable_class", "Unknown")
    class_probabilities = response_json.get("class_probabilities", [0.0] * 10)


# データフレームを作成
df_probabilities = pd.DataFrame(
    {"Digit": list(range(10)), "Probability": class_probabilities}
)

# 予測結果と確率のバーチャートを表示
st.bar_chart(df_probabilities.set_index("Digit"), height=200)
st.subheader(f"Prediction: {prediction}")
