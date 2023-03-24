import torch
import torch.nn as nn
import torch.nn.functional as Functional


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn(nsample, xyz, new_xyz):
    '''
    Input:
        xyz: [B, N, 3]
        new_xyz: [B, M, 3]
    Return:
        id: [B, nsample, M]
    '''
    # [B, N, M]
    dist = square_distance(xyz, new_xyz)
    # id: [B, nsample, M]
    id = torch.topk(dist,k=nsample,dim=1,largest=False)[1]
    id = torch.transpose(id, 1, 2)
    return id


class KNN_dist(nn.Module):
    def __init__(self,k):
        super(KNN_dist, self).__init__()
        self.R = nn.Sequential(
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,1),
        )
        self.k=k
    def forward(self,F,vertices):
        '''
        F: [B, N, dim_feats]
        vertices: [B, N, 3]
        '''
        # id: [B, N, k]
        id = knn(self.k, vertices, vertices)
        # F: [B, N, k, dim_feats]
        F = index_points(F,id)
        # v: [B, N, k, 3]
        v = index_points(vertices,id)
        # v_0: [B, N, k, 3]
        v_0 = v[:,:,0,:].unsqueeze(-2).repeat(1,1,self.k,1)
        # v_F: [B, N, k, 10]
        v_F = torch.cat((v_0, v, v_0-v, torch.norm(v_0-v,dim=-1,p=2).unsqueeze(-1)),-1)
        # v_F: [B, N, k, 1]
        v_F = self.R(v_F)
        # F: [B, N, k, dim_feats]
        F = torch.mul(v_F, F)
        # F: [B, N, dim_feats]
        F = torch.sum(F,-2)
        return F


class View_selector(nn.Module):
    def __init__(self, n_views, sampled_view, num_classes=40):
        super(View_selector, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.n_classes = num_classes
        self.cls = nn.Sequential(
            nn.Linear(512*self.s_views, 256*self.s_views),
            nn.LeakyReLU(0.2),
            nn.Linear(256*self.s_views, self.n_classes*self.s_views))
    def forward(self,F,vertices,k):
        '''
        F: [B, N, dim_feats]
        vertices: [B, N, 3]
        k: refers to `k` neighbors
        '''

        # id: [B, num_sampled_views]
        id = farthest_point_sample(vertices,self.s_views)
        # vertices1: [B, num_sampled_views, 3]
        vertices1 = index_points(vertices,id)
        # id_knn: [B, k]
        id_knn = knn(k,vertices,vertices1)
        # F: [B, N, k, dim_feats]
        F = index_points(F,id_knn)
        # vertices: [B, N]
        vertices = index_points(vertices,id_knn)
        F1 = F.transpose(1,2).reshape(F.shape[0],k,self.s_views*F.shape[-1])
        F_score = self.cls(F1).reshape(F.shape[0],k,self.s_views,self.n_classes).transpose(1,2)
        F1_ = Functional.softmax(F_score,-3)
        F1_ = torch.max(F1_,-1)[0]
        F1_id = torch.argmax(F1_,-1)
        F1_id = Functional.one_hot(F1_id,4).float()
        F1_id_v = F1_id.unsqueeze(-1).repeat(1,1,1,3)
        F1_id_F = F1_id.unsqueeze(-1).repeat(1, 1, 1, 512)
        F_new = torch.mul(F1_id_F,F).sum(-2)
        vertices_new = torch.mul(F1_id_v,vertices).sum(-2)
        return F_new,F_score,vertices_new


class LocalGCN(nn.Module):
    def __init__(self,k,n_views):
        super(LocalGCN,self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.k = k
        self.n_views = n_views
        self.KNN = KNN_dist(k=self.k)
    def forward(self,F,V):
        '''
        F: [B, N, dim_feats]
        V: [B, N, 3]
        '''

        # F: [B, N, dim_feats]
        F = self.KNN(F, V)
        # F: [B*N, dim_feats]
        F = F.view(-1, 512)
        # F: [B*N, dim_feats]
        F = self.conv(F)
        # F: [B, N, dim_feats]
        F = F.view(-1, self.n_views, 512)
        return F


class NonLocalMP(nn.Module):
    def __init__(self,n_view):
        super(NonLocalMP,self).__init__()
        self.n_view=n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Fusion = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, F):
        # F: [B, N, dim_feats]

        # F_i: [B, N, 1, dim_feats]
        F_i = torch.unsqueeze(F, 2)
        # F_j: [B, 1, N, dim_feats]
        F_j = torch.unsqueeze(F, 1)
        # F_i: [B, N, N, dim_feats]
        F_i = F_i.repeat(1, 1, self.n_view, 1)
        # F_j: [B, N, N, dim_feats]
        F_j = F_j.repeat(1, self.n_view, 1, 1)
        # M: [B, N, N, 2*dim_feats]
        M = torch.cat((F_i, F_j), 3)
        # M: [B, N, N, dim_feats]
        M = self.Relation(M)
        # M: [B, N, dim_feats]
        M = torch.sum(M,-2)
        # F: [B, N, 2*dim_feats]
        F = torch.cat((F, M), 2)
        # F: [B*N, 2*dim_feats]
        F = F.view(-1, 512 * 2)
        # F: [B*N, dim_feats]
        F = self.Fusion(F)
        # F: [B, N, dim_feats]
        F = F.view(-1, self.n_view, 512)
        return F
