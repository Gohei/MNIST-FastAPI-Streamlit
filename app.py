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
from streamlit_oekaki_component import oekaki

import requests

result = oekaki()

if result.is_submitted:
    response = requests.post(
        "http://127.0.0.1:8000/prediction", data=result.image_bytes
    )

    response_json = response.json()
    prediction = response_json.get("prediction", None)

    st.write(prediction)
