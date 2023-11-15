# MNIST-FastAPI-Streamlit

MNIST-FastAPI-Streamlitは、手書き数字認識を行うためのウェブアプリケーションです。  
MNISTデータセットでトレーニングされたモデルを使用し、StreamlitとFastAPIを使用して実際の手書き数字の識別を行うことが、このプロジェクトの主な目的です。

## 特徴
- **MNISTデータセット**を使用した手書き数字の認識
- **FastAPI**によるバックエンドAPIの構築
- **Streamlit**を使用したフロントエンドの開発

## 前提条件
このプロジェクトを実行するには以下が必要です：
- Python 3.9以上
- pipenv

## インストール
このプロジェクトの依存関係をインストールするには、以下の手順を実行してください：
```bash
pip install pipenv
```
```bash
pipenv install
```

## 使用方法
### モデルのトレーニング
```bash
pipenv run python train_model.py
```

### モデルのテスト
```bash
pipenv run python test_model.py
```

### FastAPIサーバーの起動
```bash
pipenv run python main.py
```

### Streamlitフロントエンドの起動
```bash
pipenv run streamlit run app.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.