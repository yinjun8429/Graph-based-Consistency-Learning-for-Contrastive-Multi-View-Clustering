import torch.nn as nn
from torch.nn.functional import normalize
import torch
import math
from torch.nn.parameter import Parameter
from utils import q_distribution_tool,target_distribution
import torch.nn.functional as F
from sklearn.cluster import KMeans
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.batch_size=256
        self.class_num=class_num
        self.encoders = []
        self.decoders = []
        self.weight_list=[]
        self.cluster_centers = Parameter(torch.Tensor(class_num, feature_dim*view)).to(device)
        torch.nn.init.xavier_normal_(self.cluster_centers.data)
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            # Varying the number of layers of W can obtain the representations with different shapes.
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )

        self.view = view
        self.alpha = 1.0
    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        ses=[]
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
        h = torch.cat(hs, dim=1)


        return hs, qs, xrs, zs,h

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds
    def  kl_clustering_mutil_view_loss(self,zs,z):
        # norm_squared = torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        # numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        # power = float(self.alpha + 1) / 2
        # numerator = numerator ** power
        # t_dist = (numerator.t() / torch.sum(numerator, 1)).t()  # soft assignment using t-distribution
        p=q_distribution_tool(z,self.class_num)
        p=target_distribution(p)
        q_distribution_loss=[]
        # print(len(zs))
        for v in range(self.view):
            # print(zs[v])
            q_distribution=q_distribution_tool(zs[v],self.class_num)
            # print(p)
            loss_kl = F.kl_div(q_distribution.log() ,p, reduction='batchmean')
            q_distribution_loss.append(loss_kl)
        return sum(q_distribution_loss)
    def get_assign_cluster_centers_op(self, z):
        kmeans = KMeans(self.class_num, n_init=10).fit(z.detach().cpu().numpy())
        cluster_centers_tensor = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        return cluster_centers_tensor
class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        self.batch_size = z_i.size(0)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss