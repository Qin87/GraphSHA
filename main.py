import os
import os.path as osp
import tqdm
import random
import numpy as np
import torch
import torch.nn.functional as F

from args import parse_args
from data_utils import get_dataset, get_idx_info, make_longtailed_data_remove, get_step_split, load_directedData
from gens import sampling_node_source, neighbor_sampling, duplicate_neighbor, saliency_mixup, \
    sampling_idx_individual_dst, neighbor_sampling_BiEdge, neighbor_sampling_BiEdge_bidegree, \
    neighbor_sampling_bidegree, neighbor_sampling_bidegreeOrigin, neighbor_sampling_bidegree_variant1, \
    neighbor_sampling_bidegree_variant2, neighbor_sampling_reverse
from nets import create_gcn, create_gat, create_sage
from utils import CrossEntropy
from sklearn.metrics import balanced_accuracy_score, f1_score
from neighbor_dist import get_PPR_adj, get_heat_adj, get_ins_neighbor_dist

import warnings
warnings.filterwarnings("ignore")

def train(train_idx):
    # data_x, data_y, edges, num_features, data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = load_dataset()
    global class_num_list, idx_info, prev_out
    global data_train_mask, data_val_mask, data_test_mask
    model.train()
    optimizer.zero_grad()
    if args.AugDirect == 0:
        output = model(data_x, edges[:,train_edge_mask])
        criterion(output[data_train_mask], data_y[data_train_mask]).backward()
    else:
        if epoch > args.warmup:
            # identifying source samples
            prev_out_local = prev_out[train_idx]
            sampling_src_idx, sampling_dst_idx = sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, args.tau, args.max, args.no_mask)

            if args.AugDirect == 1:
                new_edge_index = neighbor_sampling(data_x.size(0), edges[:, train_edge_mask], sampling_src_idx,
                                                   neighbor_dist_list)
            elif args.AugDirect == -1:
                new_edge_index = neighbor_sampling_reverse(data_x.size(0), edges[:, train_edge_mask], sampling_src_idx,
                                                   neighbor_dist_list)
            elif args.AugDirect == 2:
                new_edge_index = neighbor_sampling_BiEdge(data_x.size(0), edges[:, train_edge_mask],
                                                          sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 4:
                new_edge_index = neighbor_sampling_BiEdge_bidegree(data_x.size(0), edges[:, train_edge_mask],
                                                                   sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 20:
                new_edge_index = neighbor_sampling_bidegree(data_x.size(0), edges[:, train_edge_mask],
                                                            sampling_src_idx,
                                                            neighbor_dist_list)  # has two types
            elif args.AugDirect == 21:
                new_edge_index = neighbor_sampling_bidegreeOrigin(data_x.size(0), edges[:, train_edge_mask],
                                                                  sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 22:
                new_edge_index = neighbor_sampling_bidegree_variant1(data_x.size(0), edges[:, train_edge_mask],
                                                                     sampling_src_idx, neighbor_dist_list)
            elif args.AugDirect == 23:
                new_edge_index = neighbor_sampling_bidegree_variant2(data_x.size(0), edges[:, train_edge_mask],
                                                                     sampling_src_idx, neighbor_dist_list)

            else:
                raise NotImplementedError

            beta = torch.distributions.beta.Beta(1, 100)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_x = saliency_mixup(data_x, sampling_src_idx, sampling_dst_idx, lam)

        else:
            sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(class_num_list, idx_info, device)
            beta = torch.distributions.beta.Beta(2, 2)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_edge_index = duplicate_neighbor(data_x.size(0), edges[:,train_edge_mask], sampling_src_idx)
            new_x = saliency_mixup(data_x, sampling_src_idx, sampling_dst_idx, lam)

        output = model(new_x, new_edge_index)
        prev_out = (output[:data_x.size(0)]).detach().clone()
        add_num = output.shape[0] - data_train_mask.shape[0]
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device=data_x.device)
        new_train_mask = torch.cat((data_train_mask, new_train_mask), dim=0)
        sampling_src_idx = sampling_src_idx.to(torch.long).to(data_y.device)   # Ben for GPU error
        _new_y = data_y[sampling_src_idx].clone()
        new_y = torch.cat((data_y[data_train_mask], _new_y),dim =0)
        criterion(output[new_train_mask], new_y).backward()

    with torch.no_grad():
        model.eval()
        output = model(data_x, edges[:,train_edge_mask])
        val_loss= F.cross_entropy(output[data_val_mask], data_y[data_val_mask])
    optimizer.step()
    scheduler.step(val_loss)
    return num_features

@torch.no_grad()
def test():
    model.eval()
    logits = model(data_x, edges[:,train_edge_mask])
    accs, baccs, f1s = [], [], []
    for mask in [data_train_mask, data_val_mask, data_test_mask]:
        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = data_y[mask].cpu().numpy()
        acc = pred.eq(data_y[mask]).sum().item() / mask.sum().item()
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)
    return accs, baccs, f1s

def load_dataset():
    # if args.IsDirectedData:
    #     dataset = load_directedData(args)
    # else:
    #     path = args.data_path
    #     path = osp.join(path, args.undirect_dataset)
    #     dataset = get_dataset(args.undirect_dataset, path, split_type='full')
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print("Dataset is ", dataset, "\nChosen from DirectedData: ", args.IsDirectedData)

    # if os.path.isdir(log_path) is False:
    #     os.makedirs(log_path)

    data = dataset[0]
    data = data.to(device)

    global class_num_list, idx_info, prev_out, sample_times
    global data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin  # data split: train, validation, test
    try:
        data.edge_weight = torch.FloatTensor(data.edge_weight)
    except:
        data.edge_weight = None

    # if args.to_undirected:
    #     data.edge_index = to_undirected(data.edge_index)

    # copy GraphSHA
    if args.Direct_dataset.split('/')[0].startswith('dgl'):
        edges = torch.cat((data.edges()[0].unsqueeze(0), data.edges()[1].unsqueeze(0)), dim=0)
        data_y = data.ndata['label']
        data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (
            data.ndata['train_mask'].clone(), data.ndata['val_mask'].clone(), data.ndata['test_mask'].clone())
        data_x = data.ndata['feat']
        dataset_num_features = data_x.shape[1]
    # elif not args.IsDirectedData and args.undirect_dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']:
    elif not args.IsDirectedData and args.undirect_dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']:
        edges = data.edge_index  # for torch_geometric librar
        data_y = data.y
        data_x = data.x
        dataset_num_features = dataset.num_features

        data_y = data_y.long()
        n_cls = (data_y.max() - data_y.min() + 1).cpu().numpy()
        n_cls = torch.tensor(n_cls).to(device)

        train_idx, valid_idx, test_idx, train_node = get_step_split(imb_ratio=args.imb_ratio,
                                                                    valid_each=int(data.x.shape[0] * 0.1 / n_cls),
                                                                    labeling_ratio=0.1,
                                                                    all_idx=[i for i in range(data.x.shape[0])],
                                                                    all_label=data.y.cpu().detach().numpy(),
                                                                    nclass=n_cls)

        data_train_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
        data_val_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
        data_test_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
        data_train_maskOrigin[train_idx] = True
        data_val_maskOrigin[valid_idx] = True
        data_test_maskOrigin[test_idx] = True
        train_idx = data_train_maskOrigin.nonzero().squeeze()
        train_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)

        class_num_list = [len(item) for item in train_node]
        idx_info = [torch.tensor(item) for item in train_node]
    elif dataset == 'Amazon-Photo':
        pass
    else:
        edges = data.edge_index  # for torch_geometric librar
        data_y = data.y
        data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(),data.test_mask.clone())
        data_x = data.x
        try:
            dataset_num_features = dataset.num_features
        except:
            dataset_num_features = data_x.shape[1]

    # IsDirectedGraph = test_directed(edges)
    # print("This is directed graph: ", IsDirectedGraph)
    # print("data_x", data_x.shape)  # [11701, 300])

    data = data.to(device)

    data_y = data_y.long()
    # n_cls = (data_y.max() - data_y.min() + 1).cpu().numpy()
    # n_cls = torch.tensor(n_cls).to(device)
    # print("Number of classes: ", n_cls)

    return data, data_x, data_y, edges, dataset_num_features,data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin

args = parse_args()
seed = args.seed
cuda_device = args.GPUdevice
if torch.cuda.is_available():
    print("CUDA Device Index:", cuda_device)
    device = torch.device("cuda:%d" % cuda_device)
else:
    print("CUDA is not available, using CPU.")
    device = torch.device("cpu")

torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

if args.IsDirectedData:
    dataset = load_directedData(args)
else:
    path = args.data_path
    path = osp.join(path, args.undirect_dataset)
    dataset = get_dataset(args.undirect_dataset, path, split_type='full')
data, data_x, data_y, edges, num_features, data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = load_dataset()
n_cls = data_y.max().item() + 1
data = data.to(device)

# if not args.IsDirectedData and args.undirect_dataset in ['Cora', 'CiteSeer', 'PubMed']:
#     data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()
#     stats = data.y[data_train_mask]
#     n_data = []
#     for i in range(n_cls):
#         data_num = (stats == i).sum()
#         n_data.append(int(data_num.item()))
#     idx_info = get_idx_info(data.y, n_cls, data_train_mask)
#     class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
#         make_longtailed_data_remove(data.edge_index, data.y, n_data, n_cls, args.imb_ratio, data_train_mask.clone())
#     train_idx = data_train_mask.nonzero().squeeze()
#
#     labels_local = data.y.view([-1])[train_idx]
#     train_idx_list = train_idx.cpu().tolist()
#     local2global = {i:train_idx_list[i] for i in range(len(train_idx_list))}
#     global2local = dict([val, key] for key, val in local2global.items())
#     idx_info_list = [item.cpu().tolist() for item in idx_info]
#     idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in idx_info_list]
#
# elif not args.IsDirectedData and args.undirect_dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']:
#     train_idx, valid_idx, test_idx, train_node = get_step_split(imb_ratio=args.imb_ratio, \
#                                                                 valid_each=int(data_x.shape[0] * 0.1 / n_cls), \
#                                                                 labeling_ratio=0.1, \
#                                                                 all_idx=[i for i in range(data_x.shape[0])], \
#                                                                 all_label=data_y.cpu().detach().numpy(), \
#                                                                 nclass=n_cls)
#
#     data_train_mask = torch.zeros(data.x.shape[0]).bool().to(device)
#     data_val_mask = torch.zeros(data.x.shape[0]).bool().to(device)
#     data_test_mask = torch.zeros(data.x.shape[0]).bool().to(device)
#     data_train_mask[train_idx] = True
#     data_val_mask[valid_idx] = True
#     data_test_mask[test_idx] = True
#     train_idx = data_train_mask.nonzero().squeeze()
#     train_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
#
#     class_num_list = [len(item) for item in train_node]
#     idx_info = [torch.tensor(item) for item in train_node]
#
# else:
#     edges = data.edge_index  # for torch_geometric librar
#     data_y = data.y
#     data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (
#     data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())
#     data_x = data.x
#     try:
#         dataset_num_features = dataset.num_features
#     except:
#         dataset_num_features = data_x.shape[1]



if args.net == 'GCN':
    model = create_gcn(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
elif args.net == 'GAT':
    model = create_gat(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
elif args.net == "SAGE":
    model = create_sage(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
else:
    raise NotImplementedError("Not Implemented Architecture!")

model = model.to(device)
criterion = CrossEntropy().to(device)

try:
    splits = data_train_maskOrigin.shape[1]
    print("splits", splits)
    if len(data_test_maskOrigin.shape) == 1:
        data_test_maskOrigin = data_test_maskOrigin.unsqueeze(1).repeat(1, splits)
except IndexError:
    splits = 1

optimizer = torch.optim.Adam([dict(params=model.reg_params, weight_decay=5e-4), dict(params=model.non_reg_params, weight_decay=0),], lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=False)

for split in range(splits):
    if splits == 1:
        data_train_mask, data_val_mask, data_test_mask = (data_train_maskOrigin.clone(),
                                                          data_val_maskOrigin.clone(),
                                                          data_test_maskOrigin.clone())
    else:
        try:
            data_train_mask, data_val_mask, data_test_mask = (data_train_maskOrigin[:, split].clone(),
                                                              data_val_maskOrigin[:, split].clone(),
                                                              data_test_maskOrigin[:, split].clone())
        except IndexError:
            print("testIndex ,", data_test_mask.shape, data_train_mask.shape, data_val_mask.shape)
            data_train_mask, data_val_mask = (
            data_train_maskOrigin[:, split].clone(), data_val_maskOrigin[:, split].clone())
            try:
                data_test_mask = data_test_maskOrigin[:, 1].clone()
            except:
                data_test_mask = data_test_maskOrigin.clone()

    stats = data_y[data_train_mask]  # this is selected y. only train nodes of y
    n_data = []  # num of train in each class
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = get_idx_info(data_y, n_cls, data_train_mask)  # torch: all train nodes for each class
    class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
        make_longtailed_data_remove(edges, data_y, n_data, n_cls, args.imb_ratio, data_train_mask.clone())

    train_idx = data_train_mask.nonzero().squeeze()  # get the index of training data
    labels_local = data_y.view([-1])[train_idx]  # view([-1]) is "flattening" the tensor.
    train_idx_list = train_idx.cpu().tolist()
    local2global = {i: train_idx_list[i] for i in range(len(train_idx_list))}
    global2local = dict([val, key] for key, val in local2global.items())
    idx_info_list = [item.cpu().tolist() for item in idx_info]  # list of all train nodes for each class
    idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in
                      idx_info_list]  # train nodes position inside train

    if args.gdc=='ppr':
        neighbor_dist_list = get_PPR_adj(data_x, edges[:,train_edge_mask], alpha=0.05, k=128, eps=None)
    elif args.gdc=='hk':
        neighbor_dist_list = get_heat_adj(data_x, edges[:,train_edge_mask], t=5.0, k=None, eps=0.0001)
    elif args.gdc=='none':
        neighbor_dist_list = get_ins_neighbor_dist(data_y.size(0), data.edge_index[:,train_edge_mask], data_train_mask, device)

    best_val_acc_f1 = 0
    saliency, prev_out = None, None
    test_acc, test_bacc, test_f1 = 0.0, 0.0, 0.0
    for epoch in tqdm.tqdm(range(args.epoch)):
        train(train_idx)
        accs, baccs, f1s = test()
        train_acc, val_acc, tmp_test_acc = accs
        train_f1, val_f1, tmp_test_f1 = f1s
        val_acc_f1 = (val_acc + val_f1) / 2.
        if val_acc_f1 > best_val_acc_f1:
            best_val_acc_f1 = val_acc_f1
            test_acc = accs[2]
            test_bacc = baccs[2]
            test_f1 = f1s[2]
    if args.IsDirectedData:
        dataset_to_print = args.Direct_dataset
    else:
        dataset_to_print = args.undirect_dataset

    print(args.net, dataset_to_print, args.imb_ratio, "Aug:", str(args.AugDirect))
    print('acc: {:.2f}, bacc: {:.2f}, f1: {:.2f}'.format(test_acc*100, test_bacc*100, test_f1*100))

