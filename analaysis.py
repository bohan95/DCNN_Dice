import gc
import pickle
import numpy as np
import torch

def datato3d(arrays):#list of np arrays, NULL*3*100
    output=list()
    for i in arrays:
        i=np.transpose(i,(0,2,1))
        output.append(i)
    return output


"""v flip and shuffle"""
def udflip(X_nparray,y_nparray,shuffle=True):
    output=np.zeros((X_nparray.shape[0],X_nparray.shape[1],X_nparray.shape[2]),dtype=np.float32)
    for i in range(X_nparray.shape[0]):
        output[i,:]=np.flipud(X_nparray[i,:])
    output=np.vstack((X_nparray,output))
    y=np.hstack((y_nparray,y_nparray))
    if shuffle:
        shuffle_inx=np.random.permutation(output.shape[0])
        return output[shuffle_inx],y[shuffle_inx]
    else:
        return output,y

with open('../preprocessing/data_61_26.pkl','rb') as f:
    data=pickle.load(f)


dataList=datato3d([data['X_train'],data['X_test']])
X_train=dataList[0]
X_test=dataList[1]
X_train, y_train = X_train[:, :, ::5], data['y_train'][:]
X_test, y_test = X_test[:, :, ::5], data['y_test'][:]

y_test_list=data['y_test'].tolist()

NCLASS=max(y_test_list)+1

X_train,y_train=udflip(X_train,y_train,shuffle=True)
X_test,y_test=udflip(X_test,y_test,shuffle=False)

X_train=torch.from_numpy(X_train)#data['X_train'])
y_train=torch.from_numpy(y_train.astype(np.int32))#data['y_train'])

X_test=torch.from_numpy(X_test)
y_test=torch.from_numpy(y_test.astype(np.int32))  


del data,dataList
gc.collect()
print('data loaded!')
print('X_train_shape',X_train.size())
print('X_test_shape',X_test.size())
