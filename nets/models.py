import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SGConv, GATConv, APPNP, JumpingKnowledge

from nets.DiG_NoConv import DiG_SimpleXBN_nhid_Pan
from src.pgnn_conv import pGNNConv
from src.gpr_conv import GPR_prop


class pGNNNet1(torch.nn.Module):
    def __init__(self,
                 in_channels,num_hid,
                 out_channels,
                 mu=0.1,
                 p=2,
                 K=2,
                 dropout=0.5,
                 cached=False):
        super(pGNNNet1, self).__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.conv1 = pGNNConv(num_hid, out_channels, mu, p, K, cached=cached)
        self.BN1= nn.BatchNorm1d(num_hid)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.BN1(self.lin1(x)))      # Qin add BN Apr30
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

class pGNNNet2(torch.nn.Module):
    def __init__(self,
                 in_channels,num_hid,
                 out_channels,
                 mu=0.1,
                 p=2,
                 K=2,
                 dropout=0.5,
                 cached=False):
        super(pGNNNet2, self).__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.conv1 = pGNNConv(num_hid, num_hid, mu, p, K, cached=cached)
        self.conv2 = pGNNConv(num_hid, out_channels, mu, p, K, cached=cached)
        self.BN1 = nn.BatchNorm1d(num_hid)

    def forward(self, x, edge_index, edge_weight=None):
        # x = F.relu(self.lin1(x))
        x = F.relu(self.BN1(self.lin1(x)))      # Qin add BN Apr30
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

class pGNNNetX(torch.nn.Module):
    def __init__(self,
                 in_channels,num_hid,
                 out_channels,
                 mu=0.1,
                 p=2,
                 K=2,
                 dropout=0.5,layer=3,
                 cached=False):
        super(pGNNNetX, self).__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.layerx = nn.ModuleList([pGNNConv(num_hid, num_hid, mu, p, K, cached=cached) for _ in range(layer-2)])
        self.BN1 = nn.BatchNorm1d(num_hid)

        self.conv1 = pGNNConv(num_hid, out_channels, mu, p, K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.BN1(self.lin1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for iter_layer in self.layerx:
            x = F.relu(self.BN1(iter_layer(x, edge_index, edge_weight)))        # Qin add BN Apr30
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class MLPNet2(torch.nn.Module):
    def __init__(self,
                 in_channels,num_hid,
                 out_channels,
                 dropout=0.5):
        super(MLPNet2, self).__init__()
        self.dropout = dropout
        self.layer1 = torch.nn.Linear(in_channels, num_hid)
        self.layer2 = torch.nn.Linear(num_hid, out_channels)
        self.BN1 = nn.BatchNorm1d(num_hid)

    def forward(self, x, edge_index=None, edge_weight=None):
        x = torch.relu(self.BN1(self.layer1(x)))        # Qin add BN on Apr29
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x)
        return F.log_softmax(x, dim=1)

class MLPNetX(torch.nn.Module):
    def __init__(self,
                 in_channels,num_hid,
                 out_channels,
                 dropout, layer=3):
        super(MLPNetX, self).__init__()
        self.dropout = dropout
        self.layer1 = torch.nn.Linear(in_channels, num_hid)
        self.layer2 = torch.nn.Linear(num_hid, out_channels)
        self.layerx = nn.ModuleList([torch.nn.Linear(num_hid, num_hid) for _ in range(layer-2)])
        self.BN1 = nn.BatchNorm1d(num_hid)
        self.BNx = nn.BatchNorm1d(num_hid)

    def forward(self, x, edge_index=None, edge_weight=None):
        x = torch.relu(self.BN1(self.layer1(x)))    # Qin add BN Apr29
        # x = self.BN1(self.layer1(x))  # Qin add BN Apr29
        x = F.dropout(x, p=self.dropout, training=self.training)
        for iter_layer in self.layerx:
            x = F.relu(self.BNx(iter_layer(x)))    # Qin add BN Apr29
            # x = self.BNx(iter_layer(x))    # Qin add BN Apr29
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x= torch.relu(x)
        # return x
        return F.log_softmax(x, dim=1)



class MLPNet1(torch.nn.Module):
    def __init__(self,
                 in_channels,num_hid,
                 out_channels,
                 dropout=0.5):
        super(MLPNet1, self).__init__()
        self.dropout = dropout
        self.layer1 = torch.nn.Linear(in_channels, out_channels)
        # self.layer2 = torch.nn.Linear(num_hid, out_channels)

    def forward(self, x, edge_index=None, edge_weight=None):
        x = self.layer1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.layer2(x)
        return F.log_softmax(x, dim=1)

def create_MLP(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = MLPNet1(nfeat, nhid, nclass, dropout)
    elif nlayer == 2:
        model = MLPNet2(nfeat, nhid, nclass, dropout)
    else:
        model = MLPNetX(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_pan(nfeat, nhid, nclass, dropout):
    model = DiG_SimpleXBN_nhid_Pan(nfeat, nhid, nclass, dropout, layer=5)
    return model

def create_pgnn(nfeat, nhid, nclass,mu=0.1,p=2,K=2, dropout=0.5, layer=3):
    if layer == 1:
        model = pGNNNet1(nfeat, nhid, nclass,mu=0.1, p=2,K=2, dropout=0.5)
    elif layer == 2:
        model = pGNNNet2(nfeat, nhid, nclass,mu=0.1, p=2,K=2, dropout=0.5)
    else:
        model = pGNNNetX(nfeat, nhid, nclass,mu=0.1, p=2,K=2, dropout=0.5, layer=3)
    return model

def create_SGC(nfeat, nhid, nclass, dropout, nlayer, K):
    if nlayer == 1:
        model = SGCNet1(nfeat, nhid, nclass, dropout, K)
    elif nlayer == 2:
        model = SGCNet2(nfeat, nhid, nclass, dropout,K)
    else:
        model = SGCNetX(nfeat, nhid, nclass, dropout, nlayer, K)
    return model


class GCNNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid=16,
                 dropout=0.5,
                 cached=True):
        super(GCNNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, num_hid, cached=cached)
        self.conv2 = GCNConv(num_hid, out_channels, cached=cached)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GCN_Encoder(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 num_hid=16):
        super(GCN_Encoder, self).__init__()
        self.conv = GCNConv(in_channels, num_hid, cached=True)
        self.prelu = torch.nn.PReLU(num_hid)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        return x


class SGCNet1(torch.nn.Module):
    def __init__(self,
                 in_channels,nhid,
                 out_channels,dropout,
                 K=2,
                 cached=False):
        super(SGCNet1, self).__init__()
        self.dropout = dropout
        self.conv1 = SGConv(in_channels, out_channels, K=K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

class SGCNet2(torch.nn.Module):
    def __init__(self,
                 in_channels,nhid,
                 out_channels,dropout,
                 K=2,
                 cached=False):
        super(SGCNet2, self).__init__()
        self.dropout = dropout
        self.conv1 = SGConv(in_channels, nhid, K=K, cached=cached)
        self.conv2 = SGConv(nhid, out_channels, K=K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class SGCNetX(torch.nn.Module):
    def __init__(self,
                 in_channels, nhid,
                 out_channels,dropout=0.5, layer=3,
                 K=2,
                 cached=False):
        super(SGCNetX, self).__init__()
        self.dropout = dropout
        self.conv1 = SGConv(in_channels, nhid, K=K, cached=cached)
        self.conv2 = SGConv(nhid, out_channels, K=K, cached=cached)
        self.layerx = nn.ModuleList([SGConv(nhid, nhid, K=K, cached=cached) for _ in range(layer-2)])

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        for iter_layer in self.layerx:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(iter_layer(x, edge_index, edge_weight))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid=8,
                 num_heads=8,
                 dropout=0.6,
                 concat=False):

        super(GATNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, num_hid, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(num_heads * num_hid, out_channels, heads=1, concat=concat, dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=-1)


class JKNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid=16,
                 K=1,
                 alpha=0,
                 dropout=0.5,layer=4):
        super(JKNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, num_hid)
        self.conv2 = GCNConv(num_hid, num_hid)
        self.lin1 = torch.nn.Linear(num_hid, out_channels)
        self.one_step = APPNP(K=K, alpha=alpha)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=num_hid,
                                   num_layers=layer)

    def forward(self, x, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index, edge_weight)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)


class APPNPNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid=16,
                 K=1,
                 alpha=0.1,
                 dropout=0.5):
        super(APPNPNet, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.lin2 = torch.nn.Linear(num_hid, out_channels)
        self.prop1 = APPNP(K, alpha)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GPRGNNNet1(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid,
                 ppnp,
                 K=10,
                 alpha=0.1,
                 Init='PPR',
                 Gamma=None,
                 dprate=0.5,
                 dropout=0.5):
        super(GPRGNNNet1, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.lin2 = torch.nn.Linear(num_hid, out_channels)
        self.BN1 = nn.BatchNorm1d(num_hid)

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.BN1(self.lin1(x)))          # Qin add BN Apr30
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate != 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GPRGNNNet1_Qin(torch.nn.Module):
    '''
    Qin want to move prop before conv, worse
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid,
                 ppnp,
                 K=10,
                 alpha=0.1,
                 Init='PPR',
                 Gamma=None,
                 dprate=0.5,
                 dropout=0.5):
        super(GPRGNNNet1_Qin, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.lin2 = torch.nn.Linear(num_hid, out_channels)

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))


        if self.dprate != 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop1(x, edge_index, edge_weight)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class GPRGNNNet2(torch.nn.Module):
    '''
    Qin want to move prop before conv,not use
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid,
                 ppnp,
                 K=10,
                 alpha=0.1,
                 Init='PPR',
                 Gamma=None,
                 dprate=0.5,
                 dropout=0.5):
        super(GPRGNNNet2, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.lin2 = torch.nn.Linear(num_hid, out_channels)

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
            self.prop2 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)
            self.prop2 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))



        if self.dprate != 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop1(x, edge_index, edge_weight)

        if self.dprate != 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop2(x, edge_index, edge_weight)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)


        return F.log_softmax(x, dim=1)




