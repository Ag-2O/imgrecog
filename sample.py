from os import terminal_size
import pickle
import numpy as np
from matplotlib import pyplot as plt

def get_diagonal(eig_value, eig_vec, value_list):

    """ koyuuti to koyuubekutoru no kumi wo kaesu """

    for idx in range(len(eig_value)):
        value_list.append([eig_value[idx], np.hstack(eig_vec[i][idx] for i in range(len(eig_value)))])
        #print("vec_shape: {}".format(eig_vec[idx].shape))
    return value_list

def save_subspace(W):

    """ bubunn kuukann no hozonn """

    np.save("msm",W)

def load_subspace():

    """ bubunn kuukann no yomikomi """

    return np.load("msm.npy")

def adjust_data(X_train,y_train,X_test,y_test,label_list,test_label_list):

    """ 訓練データの調整 """

    train_data = []
    for ll in range(len(label_list)):
        # 訓練データは3つダブってるので
        i = ll * 3
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
    for ll in range(len(test_label_list)):
        # tesuto データ
        data = X_test[ll]

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

    return train_data, test_data

def create_subspace(train_data,dim):
    
    """ 部分関数を求める """

    W = np.empty([47,dim,dim])
    print("W.shape: {}".format(W.shape))

    #print("len_train_data: {}".format(len(train_data)))
    count = 0
    for group in train_data:
        print("count: {}".format(count))
        #print("len_group: {}".format(len(group)))
        #print("len_data: {}".format(len(data)))
        for i, data in enumerate(group):
            value_list = []
            X = data
            #print("shape_x: {}".format(X.reshape(1,-1)))
            #print("shape_x.T: {}".format(X.reshape(-1,1)))
            C = X.reshape(-1,1)*X.reshape(1,-1)
            #print("C: {}".format(C))
            eig_value, eig_vec = np.linalg.eig(C)
            #print("eig_value_size: {}, eig_vec_size: {}".format(eig_value.shape,eig_vec.shape))
            value_list = get_diagonal(eig_value, eig_vec, value_list)
            value_list = sorted(value_list,key=lambda x: x[0], reverse=True)
            #print("value_list: {}".format(value_list[1]))
            #print("sample_vec: {}, length: {}".format(value_list[1:30][1][1], len(value_list[1:30][1][1])))
            W[count,:,:] = value_list[1:100][1][1]
        #print("W.shape: {}".format(W.shape))
        count += 1
    print("部分関数を求めた！")

    return W

def run_clafic(test_data, label_list, test_label_list,W):

    """ CLAFICによる分類 """

    # yomikomi
    W = load_subspace()

    r = 100

    # bunnrui
    count_ok = 0
    count = 0
    for i, group in enumerate(test_data):
        for j, data in enumerate(group):
            out = np.empty(label_dim)
            for label in range(label_dim):
                #print("test_num: {}, label_num: {}, data_num: {}".format(i,label,j))
                # sikibetu no siki 
                out[label] = np.max((W[label,:,j].T * test_data[i][j])**2) / (np.linalg.norm(W[label,:,j].T)**2 * np.linalg.norm(test_data[i][j])**2)
                #print("W: {}".format(W[label,:,:].T))

            print("out: \n {}".format(out))
            print("test_num: {}, data_num: {}".format(i,j))
            print("arg_max_out: {}".format(np.argmax(out)))
            print("predict_label: {}".format(label_list[np.argmax(out)]))
            print("label: {}".format(test_label_list[i]))
            print("-------------------------- {} --------------------------".format(count))

            if label_list[np.argmax(out)] == test_label_list[i]:
                print("accurate!")
                count_ok += 1
            count += 1

    # 正解率
    print("accuracy: {}".format(count_ok/count))


""" main """


if __name__ == "__main__":
    # データのロード
    data = pickle.load(open("./data/ytc_py.pkl", 'rb'))
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

    # kunnrenn ラベルデータの生成
    label_list = []
    for i in range(len(y_train)):
        if y_train[i] not in label_list:
            label_list.append(y_train[i])

    label_dim = len(label_list)
    #print("label_list: {}, length: {}".format(label_list,len(label_list)))

    # test raberu de-ta no seisei
    test_label_list = []
    for i in range(len(y_test)):
        test_label_list.append(y_test[i])
    test_label_dim = len(test_label_list)

    # 画像の表示
    #plt.imshow(X_train[0][:, 10].reshape((20, 20)))
    #plt.show()

    # de-ta no tyousei
    train_data, test_data = adjust_data(X_train,y_train,X_test,y_test,label_list,test_label_list)

    try:
        # load 
        W = load_subspace()
        print("load subspace!")

    except:
        # data ga naitoki
        print("create subspace!")

        # bubunn kuukann no dousyutu
        W = create_subspace(train_data,dim)
        print("subspace created!")

        # hozonn
        save_subspace(W)
        
        # load 
        W = load_subspace()
    
    # bunnrui
    run_clafic(test_data,label_list,test_label_list,W)
