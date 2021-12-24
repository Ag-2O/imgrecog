from os import terminal_size
import pickle
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(precision=4)

def get_diagonal(eig_value, eig_vec, value_list):

    """ 固有値と固有ベクトルを返す """

    for idx in range(len(eig_value)):
        value_list.append([eig_value[idx], np.hstack(eig_vec[:,idx])])
        #print("vec_shape: {}".format(eig_vec.shape))
    return value_list


def save_subspace(W,name):

    """ 部分空間の保存 """

    np.save(name,W)


def load_subspace(name):

    """ 部分空間の読み込み """

    return np.load("py_imgrecog/subspace_default/"+ name +".npy")


def adjust_data(X_train,y_train,X_test,y_test,label_list,test_label_list):

    """ 訓練データの調整 """

    train_data = []
    for ll in range(len(label_list)):
        """
        # 訓練データは3つダブってるので
        i = ll * 3
        j = i + 1
        k = j + 1

        # データの連結
        a = X_train[i]
        b = X_train[j]
        c = X_train[k]
        data = np.hstack([a,b,c])
        """
        

        data = X_train[ll]

        # 正規化
        group_data = []
        for idx in range(data.shape[1]):
            x = data[:,idx] / np.linalg.norm(data[:,idx])
            group_data.append(x)
        train_data.append(group_data)

    """ テストデータの調整 """

    test_data = []
    for ll in range(len(test_label_list)):
        """
        # テストデータは6つダブってるので
        i = ll * 3
        j = i + 1
        k = j + 1
        l = k + 1
        m = l + 1
        n = m + 1

        # データの連結
        a = X_test[i]
        b = X_test[j]
        c = X_test[k]
        d = X_test[l]
        e = X_test[m]
        f = X_test[n]

        data = np.hstack([a,b,c,d,e,f])
        """

        data = X_test[ll]

        #print("group: {}".format(group))
        #print("length: {}".format(data.shape))

        
        # 正規化
        group_data = []
        for idx in range(data.shape[1]):
            x = data[:,idx] / np.linalg.norm(data[:,idx])
            group_data.append(x)
        test_data.append(group_data)
        
        """
        # 動作テスト用
        group_data = []
        for idx in range(10):
            x = data[:,idx] / np.linalg.norm(data[:,idx])
            group_data.append(x)
        test_data.append(group_data)
        """

    print("訓練データサイズ: {}  テストデータサイズ: {}".format(len(train_data),len(test_data)))
    return train_data, test_data


def create_subspace(train_data,dim,label_list):
    
    """ 部分空間を求める """

    W = np.empty([len(label_list),dim,dim])

    count = 0
    for g in range(len(train_data)):
        print("count: {}".format(count))
        #print("len_group: {}".format(len(group)))
        #print("len_data: {}".format(len(data)))
        R = np.empty([dim,dim])
        for i, data in enumerate(train_data[g]):
            X = data
            value_list = []
            #print("shape_x: {}".format(X.reshape(1,-1)))
            #print("shape_x.T: {}".format(X.reshape(-1,1)))
            R += X.reshape(-1,1)*X.reshape(1,-1)
            #print("C: {}".format(C))

        eig_value, eig_vec = np.linalg.eig(R)
        value_list = get_diagonal(eig_value, eig_vec, value_list)

        #print("eig_value_size: {}, eig_vec_size: {}".format(eig_value.shape,eig_vec.shape))
        #value_list = get_diagonal(eig_value, eig_vec, value_list)
        value_list = sorted(value_list,key=lambda x: x[0], reverse=True)
        
        v = np.array([x[0] for x in value_list])
        vec = np.array([x[1] for x in value_list])

        #print("value: {}".format(v))
        #print("vec: {}".format(vec))
        #print("vec.shape: {}".format(vec.shape))
        #print("value_list: {}".format(value_list))
        #print("sample_vec: {}, \n length: {}".format(value_list[1:100][1][1], len(value_list[1][1])))
        W[count,:,:] = vec
        #print("W.shape: {}".format(W.shape))
        count += 1
    print("部分空間を求めた！")
    return W


def run_clafic(test_data, label_list, test_label_list,W):

    """ CLAFICによる分類 """

    # 分類
    count_ok = 0
    count = 0
    # データセット全体
    for i, group in enumerate(test_data):
        # データセット内の1つ1つのデータ
        for j, data in enumerate(group):
            out = np.empty(len(label_list))
            # 部分空間のラベルの数分繰り返し
            for label in range(len(label_list)):
                #print("test_num: {}, label_num: {}, data_num: {}".format(i,label,j))

                # 部分空間法
                #out[label] = np.linalg.norm(W[label,:,:].T * test_data[i][j])**2

                
                S = 0
                for r in range(100):
                    S += np.dot(test_data[i][j],W[label,r,:])**2 / (np.linalg.norm(test_data[i][j]) * np.linalg.norm(W[label,r,:]))
                out[label] = S
                

                # 識別の式
                #out[label] = np.max( ((W[label,:,:].T * test_data[i][j])**2) / (np.linalg.norm(W[label,:,:].T)**2 * np.linalg.norm(test_data[i][j])**2) )
                #print("W: {}".format(W[label,:,:].T))

            print("-------------------------- {} --------------------------".format(count))
            print("out: \n {}".format(out))
            print("test_num: {}, data_num: {}".format(i,j))
            print("arg_max_out: {}".format(np.argmax(out)))
            print("predict_label: {}".format(label_list[np.argmax(out)]))
            print("label: {}".format(test_label_list[i]))

            if label_list[np.argmax(out)] == test_label_list[i]:
                print("correct!")
                count_ok += 1
            count += 1

            # 正解率
            print("accuracy: {}".format(count_ok/count))

def run_msm(test_data, label_list, test_label_list,W,Q):

    """ 相互部分空間法 """

    # 分類
    count_ok = 0
    count = 0
    # データセット全体
    for i, group in enumerate(test_data):
        # データセット内の1つ1つのデータ
        for j, data in enumerate(group):
            out = np.empty(len(label_list))
            # 部分空間のラベルの変更
            for label in range(len(label_list)):
                #print("test_num: {}, label_num: {}, data_num: {}".format(i,label,j))

                # 識別の式
                #print("W.shape: {}, Q.shape: {}".format(W[label].T.shape, Q[i].shape))
                #print("dot: {}".format(np.sum(np.dot(W[label,:,:], Q[i,:,:]))))
                #out[label] = (np.sum(np.dot(W[label,:,:], Q[i,:,:]))**2) / (np.linalg.norm(W[label,:,:])**2 * np.linalg.norm(Q[i,:,:])**2)
                #out[label] = np.max((W[label,:,:].T * Q[i,:,:])**2 / (np.linalg.norm(W[label,:,:])**2 * np.linalg.norm(Q[i,:,:])**2))
                #print("W: {}".format(W[label,:,:].T))
                
                #C = W[label,:,:].T * Q[i,:,:] * Q[i,:,:].T * W[label,:,:]
                #eig_val, eig_vec = np.linalg.eig(C)
                #out[label] = np.min(eig_val)

                
                
                S = 0
                for r in range(100):
                    S += np.dot(W[label,r,:].T, Q[i,r,:])**2 / (np.linalg.norm(W[label,r,:])**2 * np.linalg.norm(Q[i,r,:])**2)
                out[label] = S
                
                
                

            print("-------------------------- {} --------------------------".format(count))
            print("out: \n {}".format(out))
            print("test_num: {}, data_num: {}".format(i,j))
            print("arg_max_out: {}".format(np.argmax(out)))
            print("predict_label: {}".format(label_list[np.argmax(out)]))
            print("label: {}".format(test_label_list[i]))

            if label_list[np.argmax(out)] == test_label_list[i]:
                print("correct!")
                count_ok += 1
            count += 1

            # 正解率
            print("accuracy: {}".format(count_ok/count))

def run_ex_msm(test_data, label_list, test_label_list,W,Q):

    """ 改良した相互部分空間法 """

    # 分類
    count_ok = 0
    count = 0
    # データセット全体
    for i, group in enumerate(test_data):
        # データセット内の1つ1つのデータ
        for j, data in enumerate(group):
            out = np.empty(len(label_list))
            # 部分空間のラベルの変更
            for label in range(len(label_list)):
                #print("test_num: {}, label_num: {}, data_num: {}".format(i,label,j))

                # 識別の式
                #out[label] = np.max( ((W[label,:,:].T * Q[i,:,:])**2) / (np.linalg.norm(W[label,:,:].T)**2 * np.linalg.norm(Q[i,:,:])**2) )
                #print("W: {}".format(W[label,:,:].T))

                S = 0
                for r in range(100):
                    S += np.dot(W[label,r,:].T, Q[i,r,:])**2 / (np.linalg.norm(W[label,r,:].T)**2 * np.linalg.norm(Q[i,r,:])**2)
                out[label] = S

            print("-------------------------- {} --------------------------".format(count))
            print("out: \n {}".format(out))
            print("test_num: {}, data_num: {}".format(i,j))
            print("arg_max_out: {}".format(np.argmax(out)))
            print("predict_label: {}".format(label_list[np.argmax(out)]))
            print("label: {}".format(test_label_list[i]))

            if label_list[np.argmax(out)] == test_label_list[i]:
                print("correct!")
                count_ok += 1
            count += 1

    # 正解率
    print("accuracy: {}".format(count_ok/count))


def test_vector(W):

    """ 固有値と固有ベクトルを確かめる """

    W = load_subspace("msm")

    print("W.shape: {}".format(W.shape))
    print("W[1,:]: {}".format(W[1,:]))
    print("W[2,:] : {}".format(W[2,:]))


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

    # 学習のラベルデータの生成
    label_list = []
    for i in range(len(y_train)):
        label_list.append(y_train[i])
    label_dim = len(label_list)
    print("label_list_length: {}".format(len(label_list)))
    #print("label_list: {}".format(label_list))

    # テストのラベルデータの生成
    test_label_list = []
    for i in range(len(y_test)):
        test_label_list.append(y_test[i])
    test_label_dim = len(test_label_list)
    print("test_label_list_length: {}".format(len(test_label_list)))
    #print("test_label_list: {}".format(test_label_list))

    # 画像の表示
    #plt.imshow(X_train[0][:, 10].reshape((20, 20)))
    #plt.show()

    # データの調整
    train_data, test_data = adjust_data(X_train,y_train,X_test,y_test,label_list,test_label_list)

    #W = create_subspace(train_data,dim)

    try:
        # load 
        W = load_subspace("msm")
        Q = load_subspace("test")
        print("load subspace!")

    except:
        # データが無いとき
        print("create subspace!")

        # 部分空間を求める
        W = create_subspace(train_data,dim,label_list)
        save_subspace(W,"msm")

        # テストデータの部分空間
        Q = create_subspace(test_data,dim,test_label_list)
        save_subspace(Q,"test")

        print("subspace created!")
    
    #test_vector(W)

    # 分類
    #run_clafic(test_data,label_list,test_label_list,W)
    run_msm(test_data,label_list,test_label_list,W,Q)