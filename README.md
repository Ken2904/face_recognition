# face_recognition
顔判別システム
cap_face.py
名前の入力を行ってENTERを押すとカメラが立ち上がり，300枚の顔画像の収集が開始されます． 自動で顔が判別されます．

list_filename.py
収集した画像ファイルのパスを学習，テストに使用するものに分けて出力します．

Learning.py
"list_filename.py"で出力されたパスを使って顔画像データを読み込み，畳込みニューラルネットワークの学習を行い結果を出力します．

test.py
実行するとカメラが立ち上がり，　写っている人が誰であるか判別します．　畳込みニューラルネットワークの学習結果を使用します．