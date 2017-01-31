# アニメ顔分類ツール

## 使い方

1. 分類したいアニメキャラの画像をinputフォルダに格納する．(キャラごとにフォルダを作ってください．)

```
input/
    キャラA/
        1.png
        2.png
        3.png
        ...
    キャラB/
        1.png
        2.png
        3.png
    ...
```

2. 下記コマンドで顔部分を抽出する．(outputフォルダに格納されています．)
```
python detect.py
```

3. outputフォルダには分類的に間違えているデータがあるので除外する．(大事!)

4. 下記コマンドで学習させる
```
# 典型的なサンプルにあるCNNで学習
python train_cnn.py

# 最近はこれ一択的なネットワークで学習
python train_resnet.py

# 転移学習を利用して解く
python trian_resnet_plus_svm.py
```

