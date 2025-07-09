from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from scipy import sparse
from utils import normalize1

class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class RGBD(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'rgbd.npz')["view_0"].astype(np.float32)
        self.data2 = np.load(path+'rgbd.npz')["view_1"].astype(np.float32)
        self.labels = np.load(path+'rgbd.npz')["labels"].reshape(-1, 1)

    def __len__(self):
        return 1449

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
    
class LandUse_21(Dataset):
    def __init__(self, path):
        mat = scipy.io.loadmat(path + 'LandUse-21.mat')
        self.x1=sparse.csr_matrix(mat['X'][0, 0]).A.astype(np.float32)
        self.x2=sparse.csr_matrix(mat['X'][0, 1]).A.astype(np.float32)
        self.x3=sparse.csr_matrix(mat['X'][0, 2]).A.astype(np.float32)
        self.y = np.squeeze(mat['Y']).astype(np.int32).reshape(-1,1)
    def __len__(self):
        return self.x1.shape[0]
    def __getitem__(self, idx):
        # print(type(self.y))
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]),torch.from_numpy(self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
    
class Scene_15 (Dataset):
    def __init__(self, path):
        mat = scipy.io.loadmat(path + 'Scene-15.mat')
        X = mat['X'][0]
        # print(X[0].dtype)
        self.x1=X[0].astype('float32')
        self.x2=X[1].astype('float32')
  
        self.y=np.squeeze(mat['Y']).reshape(-1, 1)

    def __len__(self):
        return self.x1.shape[0]
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
class Reuters_dim10 (Dataset):
    def __init__(self, path):
        mat = scipy.io.loadmat(path + 'Reuters_dim10.mat')

        # print(X[0].dtype)
        self.x1=normalize1(np.vstack((mat['x_train'][0], mat['x_test'][0]))).astype(np.float32)
        self.x2=normalize1(np.vstack((mat['x_train'][1], mat['x_test'][1]))).astype(np.float32)
    
        self.y = np.squeeze(np.hstack((mat['y_train'], mat['y_test']))).reshape(-1, 1)

    def __len__(self):
        return self.x1.shape[0]
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "RGBD":
        dataset = RGBD('./data/')
        dims = [2048, 300]
        view = 2
        class_num = 13
        data_size = 1449
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Scene_15":
        dataset = Scene_15('./data/')
        dims = [20, 59]
        view = 2
        data_size =4485
        class_num = 15
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "Reuters_dim10":
        dataset = Reuters_dim10('./data/')
        dims = [10,10]
        view = 2
        data_size =18758
        class_num = 6
    elif dataset == "LandUse-21.mat":
        dataset = LandUse_21('./data/')
        dims = [20,59,40]
        view = 3
        data_size =2100
        class_num = 21
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
