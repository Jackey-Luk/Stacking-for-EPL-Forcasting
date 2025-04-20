from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore")

dataTrain=pd.read_csv("allAtt_onehot_large_train_new8.csv")
dataTest=pd.read_csv("allAtt_onehot_large_test_new8.csv")
print(dataTrain.head(10))
print(dataTrain.shape)




# 1. 数据加载
x_train = dataTrain.iloc[:, 6:38].values
y_train = dataTrain.iloc[:, 38:].values  # 独热编码：如 [1,0] or [0,1]
x_test = dataTest.iloc[:, 6:38].values
y_test = dataTest.iloc[:, 38:].values

# 2. Reshape 给 LSTM 用（需要 3D 输入）
x_train_lstm = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test_lstm = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# 3. 其他模型使用单标签形式
y_train_cls = np.argmax(y_train, axis=1)
y_test_cls = np.argmax(y_test, axis=1)


# 模型列表
models = {
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# 循环训练、预测与输出
for name, model in models.items():
    print(f'\n====== {name} ======')
    model.fit(x_train, y_train_cls)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test_cls, y_pred)
    print(f'Accuracy: {acc:.4f}\n')

    print(classification_report(y_test_cls, y_pred, digits=4))
