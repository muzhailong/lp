import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

path = "../dataset/data.csv"


def hex2bin(path):
    mp = dict()
    temp = [0, 0, 0, -1]
    for i in range(16):
        for j in range(len(temp) - 1, -1, -1):
            if temp[j] + 1 < 2:
                temp[j] = temp[j] + 1
                break
            else:
                temp[j] = 0
        mark = str(i)
        if i == 10:
            mark = 'A'
        elif i == 11:
            mark = 'B'
        elif i == 12:
            mark = 'C'
        elif i == 13:
            mark = 'D'
        elif i == 14:
            mark = 'E'
        elif i == 15:
            mark = 'F'
        mp[mark] = copy.deepcopy(temp)
    dp = {}

    for key1 in mp.keys():
        for key2 in mp.keys():
            if key1 + key2 not in dp:
                tmp1 = copy.deepcopy(mp[key1])
                tmp2 = copy.deepcopy(mp[key2])
                tmp1.extend(tmp2)
                if key1 == '0':
                    dp[key2] = tmp1
                dp[key1 + key2] = tmp1
    res = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip("\n")
            arr = line.split(",")
            p = []
            if arr[0] == 'T':
                p.append(1)
            else:
                p.append(0)
            for i in range(1, len(arr)):
                p.extend(dp[arr[i]])
            res.append(p)
    return np.array(res)


data = hex2bin(path)
X = data[:, 1:]
Y = data[:, 0]
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.7)
rng = np.random.RandomState(42)
unlabeled_point = rng.rand(len(train_Y)) < 0.4
train_Y[unlabeled_point] = -1

clf = LabelPropagation(n_jobs=8, gamma=0.6)
clf.fit(train_X, train_Y)

prob = clf.predict_proba(test_X)
score = roc_auc_score(test_Y, prob[:, 1])
score = round(score, 2)
fpr, tpr, _ = roc_curve(test_Y, prob[:, 1])
plt.plot(fpr, tpr, color='r')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.text(0.5, 0.4, "AUC=" + str(score), fontsize=15)
plt.show()

print(score)
