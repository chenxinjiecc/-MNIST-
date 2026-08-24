#数据加载与数据集划分
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

#去掉中文警告
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

mnist = fetch_openml("mnist_784",version = 1,cache = True,as_frame = False)
X = mnist.data / 255.0
y = mnist.target.astype(int)
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size = 0.2, random_state=42
)

sample_num=6000
X_train = X_train[:sample_num]
y_train = y_train[:sample_num]
X_test = X_test[:1500]
y_test = y_test[:1500]

print(f"训练集：{X_train.shape},测试集：{X_test.shape}")


#传统机器模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

models = {
    "KNN(k=3)":KNeighborsClassifier(n_neighbors=3),
    "KNN(k=5)": KNeighborsClassifier(n_neighbors=5),
    "KNN(k=7)": KNeighborsClassifier(n_neighbors=7),
    "SVM_linear": SVC(kernel="linear"),
    "SVM_rbf": SVC(kernel="rbf"),
    "DecisionTree": DecisionTreeClassifier(max_depth=12, random_state=42)
}
model_results = {}
for name, clf in models.items():
    print(f"正在训练 {name} ...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_results[name] = acc
    print(f"{name} 测试集准确率: {acc:.4f}")

#CNN训练
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MyData(Dataset):
    def __init__(self, data, label):
        self.x = torch.FloatTensor(data)
        self.y = torch.LongTensor(label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_ds = MyData(X_train, y_train)
test_ds = MyData(X_test, y_test)

train_load = DataLoader(train_ds, batch_size=64, shuffle=True)
test_load = DataLoader(test_ds, batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self, x):
        return self.layers(x)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(dev)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=0.001)

ep = 8
print("\n开始训练CNN模型")
for e in range(ep):
    net.train()
    loss_sum = 0
    for bx, by in train_load:
        bx, by = bx.to(dev), by.to(dev)
        out = net(bx)
        loss = loss_fn(out, by)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item()

    net.eval()
    right = 0
    with torch.no_grad():
        for bx, by in test_load:
            bx, by = bx.to(dev), by.to(dev)
            pred = net(bx).argmax(dim=1)
            right += (pred == by).sum().item()
    test_acc = right / len(test_ds)
    print(f"轮数:{e+1}, 总损失:{loss_sum:.2f}, 测试精度:{test_acc:.4f}")

model_results["CNN"] = test_acc

#生成模型准确率柱状图以及SVM_rbf 模型预测错误的样本截图
plt.figure(figsize=(9,4.5))
plt.bar(model_results.keys(), model_results.values())
plt.title("模型测试准确率对比")
plt.ylabel("准确率")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("acc.png")

svm = models["SVM_rbf"]
err_pos = (svm.predict(X_test) != y_test).nonzero()[0][:8]
plt.figure()
for i,p in enumerate(err_pos):
    plt.subplot(2,4,i+1)
    plt.imshow(X_test[p].reshape(28,28),cmap="gray")
    plt.title(f"真实:{y_test[p]},预测:{svm.predict(X_test[p:p+1])[0]}")
plt.tight_layout()
plt.savefig("error.png")

print("\n实验跑完，图片已保存")
