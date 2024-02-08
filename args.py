import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-to_undirected', '-tud', action='store_true', help='if convert graph to undirecteds')

    # parser.add_argument('--IsDirectedData', action='store_false', help='The dataset is a directed graph')
    parser.add_argument('--IsDirectedData', type=bool, default=True, help='the dataset is directed graph')
    parser.add_argument('--AugDirect', type=int, default=-1,  help='0 for noAug, 1 for one direction, 2 for bidirection aug edges, '
                                                                 '4 for bidegree and bidirection, 20 for my bidegree(best), 21 for graphSHA bidegree, 2311 is trainmask use row-degree instead of 231 use col-deg')
    parser.add_argument('--net', type=str, default='GAT', help='GNN bachbone, chosen from GCN, GAT, SAGE, APPNP, Cheb, DIG, SymDiGCN, GIN')
    parser.add_argument('--GPUdevice', type=int, default=1, help='device')
    parser.add_argument('--seed', type=int, default=100, help='seed')
    parser.add_argument('--undirect_dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed', 'Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS'], default='Cora', help='dataset name')
    parser.add_argument('--Direct_dataset', type=str, default='WebKB/texas', help='dgl/cora, dgl/citeseer, dgl/pubmed..., citeseer_npz/ , cora_ml/,  WikiCS/, '
                                                                              ' WikipediaNetwork/squirrel, WikipediaNetwork/chameleon '
                                                                              'WebKB/Cornell, WebKB/texas, WebKB/wisconsin')
    # parser.add_argument('--data_path', type=str, default='datasets/', help='data path')
    parser.add_argument('--imb_ratio', type=float, default=100, help='imbalance ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout prob')
    parser.add_argument('--num_filter', type=int, default=2, help='num of filters')
    parser.add_argument('--layer', type=int, default=2, help='number of layers (2 or 3), default: 2')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha teleport prob')
    parser.add_argument('-K', '--K', default=2, type=int)  # for cheb

    parser.add_argument('--feat_dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--warmup', type=int, default=5, help='warmup epoch')
    parser.add_argument('--epoch', type=int, default=1500, help='epoch')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    # parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')
    parser.add_argument('--tau', type=int, default=2, help='temperature in the softmax function when calculating confidence-based node hardness')
    parser.add_argument('--max', action="store_true", help='synthesizing to max or mean num of training set. default is mean') 
    parser.add_argument('--no_mask', action="store_true", help='whether to mask the self class in sampling neighbor classes. default is mask')
    parser.add_argument('--gdc', type=str, choices=['ppr', 'hk', 'none'], default='ppr', help='how to convert to weighted graph')

    parser.add_argument('-hds', '--heads', default=8, type=int)
    parser.add_argument('--log_root', type=str, default='../logs/',
                        help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test',
                        help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/',
                        help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')

    args = parser.parse_args()

    return args
