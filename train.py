import torch
from network import Network,InstanceLoss
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from utils import *
import torch.nn as nn
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# LandUse-21.mat
# Reuters_dim10
# RGBD
Dataname = 'Caltech-5V'
print(Dataname)
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--batch_size_cl', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0005)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=30)
parser.add_argument("--tune_epochs", default=1)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument('--lambda1', default=0.01, type=float)
parser.add_argument('--lambda2', default=0.01, type=float)
parser.add_argument('--lambda3', default=0.01, type=float)
parser.add_argument('--lambda4', default=0.1, type=float)
parser.add_argument('--class_num', default=1, type=float)
parser.add_argument('--CL_temperature', default=1, type=float)
parser.add_argument('--knn', default=10, type=int, help='number of nodes for subgraph embedding')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "MNIST-USPS":
    args.con_epochs = 100
    seed = 10
if args.dataset == "BDGP":
    args.con_epochs = 100
    seed = 10
if args.dataset == "CCV":
    args.con_epochs = 100
    seed = 1
if args.dataset == "Fashion":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Caltech-2V":
    args.class_num=7
    args.con_epochs = 100
    seed = 10
if args.dataset == "Caltech-3V":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Caltech-4V":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Caltech-5V":
    args.con_epochs = 100
    seed = 10
if args.dataset == "RGBD":
    args.con_epochs = 150
    seed = 10
if args.dataset == "Scene_15":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Reuters_dim10":
    args.con_epochs = 100
    seed = 10
if args.dataset == "LandUse-21.mat":
    args.con_epochs = 100
    seed = 3
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)





def pretrain1(epoch):
    print("------------------pretrain--------------------")
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, zs,z = model(xs)
        # model.cluster_centers.data = model.get_assign_cluster_centers_op(z).to(device)
        # cl_graph_loss, con_graph_loss = graph_contrastive_train(epoch, xs, zs)
        loss_list = []
        loss_con=[]
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
def pretrain(epoch):
    print("------------------pretrain--------------------")
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, zs,z = model(xs)
        if epoch>100:
            cl_graph_loss,con_graph_loss=graph_contrastive_train(epoch,xs,zs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss=sum(loss_list)
        if epoch>100:
            # kl_loss=model.kl_clustering_mutil_view_loss(zs,z)
            loss=loss+args.lambda1*cl_graph_loss+args.lambda2*con_graph_loss
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(loss))

def graph_contrastive_train(epoch,xs,zs):
    cl_loss_all=0.
    con_loss_all=0.
    mes = torch.nn.MSELoss()
    x_knn=[]
    for v in range(view):
        if epoch<120:
            indices = get_k_nearest_neighbors(xs[v].cpu())
        else:
            indices = get_k_nearest_neighbors(zs[v].detach().cpu().numpy())
        x0_knn = zs[v][indices]
        x0_knn = x0_knn.reshape(x0_knn.shape[0] * x0_knn.shape[1], x0_knn.shape[2])
        x_knn.append(x0_knn)
    for v in range(view):
        cl_loss_all+=criterion_instance(zs[v],zs[v])
    for i in range(args.knn):
        num_i = np.arange(256) * args.knn + i
        for v in range(view):
            cl_loss_all+= criterion_instance(x_knn[v][num_i], zs[v]) / args.knn
    for j in range(256):
        num_j = np.arange(args.knn) + j * args.knn
        for v in range(view):
            for w in range(v + 1, view):
                con_loss_all+= mes(x_knn[v][num_j], x_knn[w][num_j])
    return cl_loss_all,con_loss_all



def contrastive_train(epoch,loss_print):
    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs,z= model(xs)
        cl_graph_loss,con_graph_loss=graph_contrastive_train(epoch,xs,zs)
        loss_list = []
        loss_list2=[]
        # kl_loss=model.kl_clustering_mutil_view_loss(zs,z)
        # for v in range(view):
        #     for w in range(v+1, view):
        #         loss_list.append(criterion.forward_feature(hs[v], hs[w]))
        #         loss_list.append(criterion.forward_label(qs[v], qs[w]))
        #     loss_list2.append(mes(xs[v], xrs[v]))
        # loss1=sum(loss_list)
        # loss=loss1
        # loss = 0*loss1+args.lambda1*cl_graph_loss+args.lambda2*con_graph_loss+0*sum(loss_list2)
        loss=cl_graph_loss+con_graph_loss
        if epoch>100:
            q_global=q_distribution_tool(z,7)
        
        # q = q_global.detach().cpu()
        # q = torch.argmax(q, dim=1).numpy()
        # kl_loss=model.kl_clustering_mutil_view_loss(zs,z)
            for v in range(view):
                p = q_global.detach().cpu()
                p = torch.argmax(p, dim=1).numpy()
                with torch.no_grad():
                    q = qs[v].detach().cpu()
                    q = torch.argmax(q, dim=1).numpy()
                    p_hat = match(p, q)

                loss_list2.append(cross_entropy(qs[v], p_hat))
        loss_clu=0.1*sum(loss_list2)
        loss = 0.1*sum(loss_list2)+loss
        
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    loss_print.append(loss.item())
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(loss))
    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(loss),'Loss_G:{:.6f}'.format(args.lambda1*cl_graph_loss+args.lambda2*con_graph_loss),'Loss_MF:{:.6f}'.format(loss1),'loss_clu{:.6f}'.format(loss_clu))


def make_pseudo_label(model, device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.eval()
    scaler = MinMaxScaler()
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _,_ = model.forward(xs)
        for v in range(view):
            hs[v] = hs[v].cpu().detach().numpy()
            hs[v] = scaler.fit_transform(hs[v])

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []
    for v in range(view):
        Pseudo_label = kmeans.fit_predict(hs[v])
        Pseudo_label = Pseudo_label.reshape(data_size, 1)
        Pseudo_label = torch.from_numpy(Pseudo_label)
        new_pseudo_label.append(Pseudo_label)

    return new_pseudo_label


def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y

def fine_tuning1(epoch, new_pseudo_label):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_idx, (xs, _, idx) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, qs, _, _,z = model(xs)
        # p_global=q_distribution_tool(z,args.class_num)
        loss_list = []
        for v in range(view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                p_hat = match(p, q)
            loss_list.append(cross_entropy(qs[v], p_hat))
            # loss_list.append(cross_entropy(qs[v],p_global))
            # loss_list.append(cross_entropy(p_hat, p_global.float()))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
def fine_tuning(epoch, new_pseudo_label):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    # kl_loss=model.kl_clustering_mutil_view_loss(zs,z)
    for batch_idx, (xs, _, idx) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, qs, _, zs,z = model(xs)
        q_global=q_distribution_tool(z,7)
        loss_list = []
        # q = q_global.detach().cpu()
        # q = torch.argmax(q, dim=1).numpy()
        # kl_loss=model.kl_clustering_mutil_view_loss(zs,z)
        for v in range(view):
            p = q_global.detach().cpu()
            p = torch.argmax(p, dim=1).numpy()
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                
                
                p_hat = match(p, q)

            loss_list.append(cross_entropy(qs[v], p_hat))
            # loss_list.append(cross_entropy(qs[v], q_global))
            

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
data_loader_cl = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_cl,
        shuffle=True,
        drop_last=True,
    )

accs = []
nmis = []
purs = []
accs1 = []
nmis1 = []
purs1 = []
loss_print=[]
acc_print = []
nmi_print= []
pur_print= []
if not os.path.exists('./models'):
    os.makedirs('./models')


 
T = 1
for i in range(1):
    print(Dataname)
    print("ROUND:{}".format(i+1))
    criterion_instance = InstanceLoss(args.batch_size, args.CL_temperature, 0).to(device)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    # print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    while epoch <= args.mse_epochs + args.con_epochs:
        contrastive_train(epoch,loss_print)
       
        if epoch%1==0:
            acc1,nmi1,pur1, acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num,epoch, eval_h=False)
            acc_print.append(acc)
            nmi_print.append(nmi)
            pur_print.append(pur)
        if epoch == args.mse_epochs + args.con_epochs:
            print(loss_print)
            acc1,nmi1,pur1, acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num,epoch,eval_h=False)
        epoch += 1
    new_pseudo_label = make_pseudo_label(model, device)
    while epoch <= args.mse_epochs + args.con_epochs + args.tune_epochs:
        fine_tuning(epoch, new_pseudo_label)
        if epoch == args.mse_epochs + args.con_epochs + args.tune_epochs:
            acc1,nmi1,pur1, acc, nmi, pur= valid(model, device, dataset, view, data_size, class_num,epoch,eval_h=False)
        # state = model.state_dict()
        # torch.save(state, './models/' + args.dataset + '.pth')
        # print('Saving..')
            accs.append(acc)
            nmis.append(nmi)
            purs.append(pur)
            accs1.append(acc1)
            nmis1.append(nmi1)
            purs1.append(pur1)
        epoch += 1
print("Clustering results on global features: ")
print(accs1)
print(nmis1)
print(purs1)
print("Clustering results on semantic labels: ")
print(accs)
print(nmis)
print(purs)
print(acc_print)
print(nmi_print)
print(pur_print)