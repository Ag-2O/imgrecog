from os import terminal_size
import pickle
import numpy as np
from matplotlib import pyplot as plt

def get_diagonal(eig_value, eig_vec, value_list):
    for idx in range(len(eig_value)):
        value_list.append([eig_value[idx], eig_vec[idx]])
    return value_list

# github

# データのロード
data = pickle.load(open("ytc_py.pkl", 'rb'))
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

# データの情報
print("Number of training image-sets: ", len(X_train))
print("Number of testing image-sets: ", len(X_test))
print("Feature dimension of each image: ", X_train[0].shape[0])

# データサイズ
train_size = len(X_train)
test_size = len(X_test)
dim = X_train[0].shape[0]

# ラベルデータの生成
label_list = []
for i in range(len(y_train)):
    if y_train[i] not in label_list:
        label_list.append(y_train[i])

label_dim = len(label_list)
#print("label_list: {}, length: {}".format(label_list,len(label_list)))

# 画像の表示
#plt.imshow(X_train[0][:, 10].reshape((20, 20)))
#plt.show()

""" 訓練データの調整 """

train_data = []
for group in range(len(label_list)):
    # 訓練データは3つダブってるので
    i = group * 3
    j = i + 1
    k = j + 1

    # データの連結
    a = X_train[i]
    b = X_train[j]
    c = X_train[k]
    data = np.hstack([a,b,c])

    #print("group: {}".format(group))
    #print("length: {}".format(data.shape))

    # 正規化
    group_data = []
    for idx in range(data.shape[1]):
        x = data[:,idx]
        x_l2_norm = sum(x**2)**0.5
        x_l2_normalized = x / x_l2_norm
        group_data.append(x_l2_normalized)
    train_data.append(group_data)
print("訓練データを調整した！")

""" テストデータの調整 """

test_data = []
for group in range(len(label_list)):
    # 訓練データは3つダブってるので
    i = group * 3
    j = i + 1
    k = j + 1

    # データの連結
    a = X_train[i]
    b = X_train[j]
    c = X_train[k]
    data = np.hstack([a,b,c])

    #print("group: {}".format(group))
    #print("length: {}".format(data.shape))

    # 正規化
    group_data = []
    for idx in range(data.shape[1]):
        x = data[:,idx]
        x_l2_norm = sum(x**2)**0.5
        x_l2_normalized = x / x_l2_norm
        group_data.append(x_l2_normalized)
    test_data.append(group_data)
print("テストデータを調整した！")

""" 部分関数を求める """

W = np.empty([47,dim,dim])

#print("len_train_data: {}".format(len(train_data)))
count = 0
for group in train_data:
    print("count: {}".format(count))
    #print("len_group: {}".format(len(group)))
    value_list = []
    #print("len_data: {}".format(len(data)))
    for data in group:
        X = data
        #print("shape_x: {}".format(X.reshape(1,-1)))
        #print("shape_x.T: {}".format(X.reshape(-1,1)))
        C = X.reshape(1,-1)*X.reshape(-1,1)
        #print("C: {}".format(C))
        eig_value, eig_vec = np.linalg.eig(C)
        value_list = get_diagonal(eig_value, eig_vec, value_list)
    
    value_list = sorted(value_list,key=lambda x: x[0], reverse=True)
    #print("value_list: {}".format(value_list[1]))
    W[count,:,:] = value_list[1][1:100]
    count += 1
print("部分関数を求めた！")

""" CLAFICによる分類 """

out = np.empty(label_dim)
count_ok = 0
for i in range(test_size):
    out = np.empty(label_dim)
    for j in range(label_dim):
        norm = np.dot(test_data[i][j], W[j, :, k])**2
        out[j] = norm
    if np.argmax(out) == j:
        count_ok += 1

# 正解率
print("accuracy: {}".format(count_ok/test_size * 100))
