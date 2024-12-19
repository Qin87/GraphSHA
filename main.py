################################
# compared to All2Main.py, this version to ensure that when I stop the process half way, it still could print the result.
################################
import signal
import statistics
import sys
import time

import random
import numpy as np
import torch
import torch.nn.functional as F

from args import parse_args
from data_utils import get_idx_info, make_longtailed_data_remove, keep_all_data
from edge_nets.Edge_DiG_ import edge_prediction
from edge_nets.edge_data import get_appr_directed_adj, get_second_directed_adj, get_second_directed_adj_union, get_third_directed_adj_union, get_4th_directed_adj, \
    get_4th_directed_adj_union, WCJ_get_directed_adj, Qin_get_second_directed_adj, Qin_get_directed_adj, get_appr_directed_adj2
from data_model import CreatModel, load_dataset, log_file, get_name
from nets.src2 import laplacian
from nets.src2.quaternion_laplacian import process_quaternion_laplacian
from preprocess import  F_in_out,  F_in_out_Qin,  F_in_out0
from utils import CrossEntropy
from sklearn.metrics import balanced_accuracy_score, f1_score

import warnings

from torch_geometric.data import Data
warnings.filterwarnings("ignore")


def signal_handler(sig, frame):
    global end_time
    end_time = time.time()
    print("Process interrupted!")
    # calculate_time()
    log_results()
    sys.exit(0)

def log_results():
    global start_time, end_time
    if start_time is not None and end_time is not None:
        with open(log_directory + log_file_name_with_timestamp, 'a') as log_file:
            elapsed_time = end_time - start_time
            print("Total time: {:.2f} seconds".format(elapsed_time), file=log_file)
            print("Total time: {:.2f} seconds".format(elapsed_time))
            if len(macro_F1) > 1:
                average = statistics.mean(macro_F1)
                std_dev = statistics.stdev(macro_F1)
                average_acc = statistics.mean(acc_list)
                std_dev_acc = statistics.stdev(acc_list)
                average_bacc = statistics.mean(bacc_list)
                std_dev_bacc = statistics.stdev(bacc_list)
                print(net_to_print + str(args.layer) + '_'+dataset_to_print + "_Aug" + str(args.AugDirect) + "_acc" + f"{average_acc:.1f}±{std_dev_acc:.1f}" + "_bacc" + f"{average_bacc:.1f}±{std_dev_bacc:.1f}" + '_MacroF1:' + f"{average:.1f}±{std_dev:.1f},{len(macro_F1):2d}splits")
                print(net_to_print + str(args.layer) + '_'+dataset_to_print + "_Aug" + str(args.AugDirect) + "_acc" + f"{average_acc:.1f}±{std_dev_acc:.1f}" + "_bacc" + f"{average_bacc:.1f}±{std_dev_bacc:.1f}" + '_MacroF1:' + f"{average:.1f}±{std_dev:.1f},{len(macro_F1):2d}splits", file=log_file)
            elif len(macro_F1) == 1:
                print(net_to_print+str(args.layer)+'_'+dataset_to_print +"_Aug"+str(args.AugDirect)+"_acc"+f"{acc_list[0]:.1f}"+"_bacc" + f"{bacc_list[0]:.1f}"+'_MacroF1_'+f"{macro_F1[0]:.1f}, 1split", file=log_file)
                print(net_to_print+str(args.layer)+'_'+dataset_to_print +"_Aug"+str(args.AugDirect)+"_acc"+f"{acc_list[0]:.1f}"+"_bacc" + f"{bacc_list[0]:.1f}"+'_MacroF1_'+f"{macro_F1[0]:.1f}, 1split")
            else:
                print("not a single split is finished")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def train(train_idx, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight, X_real, X_img, Sigedge_index, norm_real, norm_imag,
          X_img_i, X_img_j, X_img_k,norm_img_i,norm_img_j, norm_img_k, Quaedge_index):
    # print("come to train")

    global class_num_list, idx_info, prev_out
    global data_train_mask, data_val_mask, data_test_mask
    new_edge_index=None
    new_x = None
    new_y = None
    new_y_train = None
    model.train()

    optimizer.zero_grad()
    if args.net.startswith(('Sym', 'addSym', 'Qym', 'addQym')):
        out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight)
        # out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight, edge_Qin_in_tensor, edge_Qin_out_tensor)
    elif args.net.startswith(('Di', 'Qi', 'Wi')):
        if args.net[3:].startswith(('Sym', 'Qym')):
            # out = model(new_x, new_edge_index, edge_in, in_weight, edge_out, out_weight, new_SparseEdges, edge_weight)
            out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight,SparseEdges, edge_weight)
        else:
            out = model(data_x, SparseEdges, edge_weight)
    elif args.net.startswith('Mag'):
        out = model(X_real, X_img, edges, args.q, edge_weight)  # (1,5,183)
        # out = out.permute(2, 1, 0).squeeze()        # (183,5)
    elif args.net.startswith('Sig'):
        out = model(X_real, X_img, norm_real, norm_imag, Sigedge_index)
    elif args.net.startswith('Qua'):
        out = model(X_real, X_img_i, X_img_j, X_img_k,norm_img_i, norm_img_j, norm_img_k, norm_real,Quaedge_index)
    else:
        out = model(data_x, edges)
    criterion(out[data_train_mask], data_y[data_train_mask]).backward()

    with torch.no_grad():
        model.eval()
        if args.net.startswith(('Sym', 'addSym', 'Qym', 'addQym')):
            out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight)

        elif args.net.startswith(('Di', 'Qi', 'Wi')):

            if args.net[3:].startswith(('Sym', 'Qym')):
                out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight)
            else:
                out = model(data_x, SparseEdges, edge_weight)
        elif args.net.startswith('Mag'):
            out = model(X_real, X_img, edges, args.q, edge_weight)
        elif args.net.startswith('Sig'):
            out = model(X_real, X_img, norm_real, norm_imag, Sigedge_index)
        elif args.net.startswith('Qua'):  #
            out = model(X_real, X_img_i, X_img_j, X_img_k,norm_img_i, norm_img_j, norm_img_k, norm_real,Quaedge_index)
        else:
            out = model(data_x, edges)
        val_loss = F.cross_entropy(out[data_val_mask], data_y[data_val_mask])
    optimizer.step()
    scheduler.step(val_loss, epoch)

    return val_loss, new_edge_index, new_x, new_y, new_y_train

@torch.no_grad()
def test():
    global edge_in, in_weight, edge_out, out_weight

    model.eval()

    if args.net.startswith(('Sym', 'addSym', 'Qym', 'addQym')):
        logits = model(data_x, edges[:, train_edge_mask], edge_in, in_weight, edge_out, out_weight)
    elif args.net.startswith(('Di', 'Qi', 'Wi')):
        if args.net[3:].startswith(('Sym', 'Qym')):
            logits = model(data_x, edges, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight)
        else:
            logits = model(data_x, SparseEdges, edge_weight)
    elif args.net.startswith('Mag'):
        logits = model(X_real, X_img, edges, args.q, edge_weight)
    elif args.net.startswith('Sig'):  #
        logits = model(X_real, X_img, norm_real, norm_imag, Sigedge_index)
    elif args.net.startswith('Qua'):  #
        logits = model(X_real, X_img_i, X_img_j, X_img_k, norm_imag_i, norm_imag_j, norm_imag_k, norm_real, Quaedge_index)
    else:
        logits = model(data_x, edges[:, train_edge_mask])
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


start_time = time.time()
args = parse_args()
seed = args.seed
cuda_device = args.GPUdevice
if torch.cuda.is_available():
    print("cuda Device Index:", cuda_device)
    device = torch.device("cuda:%d" % cuda_device)
else:
    print("cuda is not available, using CPU.")
    device = torch.device("cpu")
if args.IsDirectedData and args.Direct_dataset.split('/')[0].startswith('dgl'):
    device = torch.device("cpu")
    print("dgl, using CPU.")
if args.CPU:
    device = torch.device("cpu")
    print("args.CPU true, using CPU.")

net_to_print, dataset_to_print = get_name(args)

log_directory, log_file_name_with_timestamp = log_file(net_to_print, dataset_to_print, args)
print(args)
with open(log_directory + log_file_name_with_timestamp, 'w') as log_file:
    print(args, file=log_file)

torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.qinchmark = False
random.seed(seed)
np.random.seed(seed)

edge_in = None
in_weight = None
edge_out = None
out_weight = None
SparseEdges = None
edge_weight = None
X_img = None
X_real = None
edge_Qin_in_tensor = None
edge_Qin_out_tensor = None
Sigedge_index = None
norm_real = None
norm_imag = None
norm_imag_i =None
norm_imag_j = None
norm_imag_k = None
Quaedge_index = None
X_img_i = None
X_img_j = None
X_img_k = None

gcn = True

pos_edges = None
neg_edges = None

macro_F1 = []
acc_list = []
bacc_list = []

criterion = CrossEntropy().to(device)

data, data_x, data_y, edges, num_features, data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = load_dataset(args, device)
n_cls = data_y.max().item() + 1
print("class number is ", n_cls)
if args.net.startswith('Di'):
    edge_index1, edge_weights1 = get_appr_directed_adj2(args.alpha, edges.long(), data_y.size(-1), data_x.dtype)    # consumiing for large graph
    if args.net[-1].isdigit() and (args.net[-2] == 'i' or args.net[-2] == 'u'):
        k = int(args.net[-1])
        if args.net[-2] == 'i':
            edge_index_tuple, edge_weights_tuple = get_second_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
        else:
            edge_index_tuple, edge_weights_tuple = get_second_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype, k)
        SparseEdges = (edge_index1, edge_index_tuple)
        edge_weight = (edge_weights1, edge_weights_tuple)
        del edge_index_tuple, edge_weights_tuple
    else:
        SparseEdges = edge_index1
        edge_weight = edge_weights1
    del edge_index1, edge_weights1

    if args.net[3:].startswith(('Sym', 'Qym')):
        if args.net[3:].startswith('Qym'):
            data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edges.long(), data_y.size(-1), data.edge_weight)
        else:
            data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out0(edges.long(), data_y.size(-1), data.edge_weight)

elif args.net.startswith(('Qi', 'Wi', 'pan')):
    if args.net.startswith('Wi'):
        edge_index1, edge_weights1 = WCJ_get_directed_adj(args.alpha, edges.long(), data_y.size(-1), data_x.dtype, args.W_degree)
    else:
        edge_index1, edge_weights1 = Qin_get_directed_adj(args.alpha, edges.long(), data_y.size(-1), data_x.dtype)
    if args.net[-1].isdigit() and (args.net[-2] == 'i' or args.net[-2] == 'u'):
        k = int(args.net[-1])
        if args.net[-2] == 'i':
            edge_index_tuple, edge_weights_tuple = Qin_get_second_directed_adj(edges.long(), data_y.size(-1), data_x.dtype, k)
        else:
            edge_index_tuple, edge_weights_tuple = get_second_directed_adj_union(edges.long(), data_y.size(-1), data_x.dtype, k)
        SparseEdges = (edge_index1,) + edge_index_tuple
        edge_weight = (edge_weights1,) + edge_weights_tuple

        del edge_index_tuple, edge_weights_tuple
    else:
        SparseEdges = edge_index1
        edge_weight = edge_weights1
    del edge_index1, edge_weights1
    if args.net[3:].startswith('Qym'):
        data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edges.long(), data_y.size(-1), data.edge_weight)
    elif args.net[3:].startswith('Sym'):
        data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out0(edges.long(), data_y.size(-1), data.edge_weight)

elif args.net.startswith(('Sym', 'addSym', 'Qym', 'addQym')):
    if args.net.startswith(('Qym', 'addQym')):
        data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edges.long(),data_y.size(-1),data.edge_weight)
    else:
        data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out0(edges.long(),data_y.size(-1),data.edge_weight)
elif args.net.startswith(('Mag', 'Sig', 'Qua')):
    data_x_cpu = data_x.cpu()
    X_img_i = torch.FloatTensor(data_x_cpu).to(device)
    X_img_j = torch.FloatTensor(data_x_cpu).to(device)
    X_img_k = torch.FloatTensor(data_x_cpu).to(device)
    X_img = torch.FloatTensor(data_x_cpu).to(device)
    X_real = torch.FloatTensor(data_x_cpu).to(device)
    if args.net.startswith('Sig'):
        Sigedge_index, norm_real, norm_imag = laplacian.process_magnetic_laplacian(edge_index=edges, gcn=gcn, net_flow=args.netflow, x_real=X_real, edge_weight=edge_weight,
                                                                                   normalization='sym', return_lambda_max=False)
    elif args.net.startswith('Qua'):
        Quaedge_index, norm_real, norm_imag_i, norm_imag_j, norm_imag_k = process_quaternion_laplacian(edge_index=edges, x_real=X_real, edge_weight=edge_weight,
                                                                                                    normalization='sym', return_lambda_max=False)

else:
    pass

try:
    splits = data_train_maskOrigin.shape[1]
    print("splits", splits)
    if len(data_test_maskOrigin.shape) == 1:
        data_test_maskOrigin = data_test_maskOrigin.unsqueeze(1).repeat(1, splits)
except:
    splits = 1
Set_exit = False
try:
    # start_time = time.time()
    with open(log_directory + log_file_name_with_timestamp, 'a') as log_file:
        print('Using Device: ',device, file=log_file)
        # for split in range(splits - 1, -1, -1):
        for split in range(splits):
            model = CreatModel(args, num_features, n_cls, data_x, device).to(device)
            if split==0:
                print(model, file=log_file)
                print(model)
            # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
            if hasattr(model, 'coefs'):     # parameter without weight_decay will typically change faster
                optimizer = torch.optim.Adam(
                    [dict(params=model.reg_params, lr=args.lr, weight_decay=5e-4), dict(params=model.non_reg_params, lr=args.lr, weight_decay=0),
                     dict(params=model.coefs, lr=args.coeflr * args.lr, weight_decay=args.wd4coef), ],
                )
            elif hasattr(model, 'reg_params'):
                optimizer = torch.optim.Adam(
                    [dict(params=model.reg_params, weight_decay=5e-4), dict(params=model.non_reg_params, weight_decay=0), ], lr=args.lr)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)


            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=80, verbose=True)

            if splits == 1:
                data_train_mask, data_val_mask, data_test_mask = (data_train_maskOrigin.clone(),data_val_maskOrigin.clone(),data_test_maskOrigin.clone())
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
            idx_info = get_idx_info(data_y, n_cls, data_train_mask, device)  # torch: all train nodes for each class
            node_train = torch.sum(data_train_mask).item()

            class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
                keep_all_data(edges, data_y, n_data, n_cls,  data_train_mask)
            print(dataset_to_print + '\ttotalNode_' + str(data_train_mask.size()[0]) + '\t trainNodeBal_' + str(node_train) + '\t trainNodeNow_' + str(torch.sum(
                data_train_mask).item()), file = log_file)
            print(dataset_to_print + '\ttotalEdge_' + str(edges.size()[1]) + '\t trainEdgeBal_' + str(train_edge_mask.size()[0]) + '\t trainEdgeNow_' + str(
                torch.sum(train_edge_mask).item()), file = log_file)
            print(dataset_to_print + '\ttotalNode_' + str(data_train_mask.size()[0]) + '\t trainNodeBal_' + str(node_train) + '\t trainNodeNow_' + str(torch.sum(
                data_train_mask).item()))
            print(dataset_to_print + '\ttotalEdge_' + str(edges.size()[1]) + '\t trainEdgeBal_' + str(train_edge_mask.size()[0]) + '\t trainEdgeNow_' + str(
                torch.sum(train_edge_mask).item()))
            sorted_list = sorted(class_num_list, reverse=True)
            sorted_list_original = sorted(n_data, reverse=True)

            print('original class_num_list is ', sorted_list_original)
            print('New      class_num_list is ', sorted_list)
            if sorted_list[-1]:
                imbalance_ratio_origin = sorted_list_original[0] / sorted_list_original[-1]
                print('Origin Imbalance ratio is {:.1f}'.format(imbalance_ratio_origin))
                imbalance_ratio = sorted_list[0] / sorted_list[-1]
                print('New    Imbalance ratio is {:.1f}'.format(imbalance_ratio))
            else:
                print('the minor class has no training sample')
            train_idx = data_train_mask.nonzero().squeeze()  # get the index of training data
            val_idx = data_val_mask.nonzero().squeeze()  # get the index of training data
            test_idx = data_test_mask.nonzero().squeeze()  # get the index of training data
            labels_local = data_y.view([-1])[train_idx]  # view([-1]) is "flattening" the tensor.
            train_idx_list = train_idx.cpu().tolist()
            local2global = {i: train_idx_list[i] for i in range(len(train_idx_list))}
            global2local = dict([val, key] for key, val in local2global.items())
            idx_info_list = [item.cpu().tolist() for item in idx_info]  # list of all train nodes for each class
            idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in
                              idx_info_list]  # train nodes position inside train

            best_val_acc_f1 = 0
            best_val_f1 = 0
            best_val_acc = 0
            saliency, prev_out = None, None
            test_acc, test_bacc, test_f1 = 0.0, 0.0, 0.0
            CountNotImproved = 0
            end_epoch = 0
            # for epoch in tqdm.tqdm(range(args.epoch)):
            for epoch in range(args.epoch):
                val_loss, new_edge_index, new_x, new_y, new_y_train = train(train_idx, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight, X_real, X_img, Sigedge_index, norm_real,norm_imag,
                                                                                X_img_i, X_img_j, X_img_k,norm_imag_i, norm_imag_j, norm_imag_k, Quaedge_index)
                accs, baccs, f1s = test()
                train_acc, val_acc, tmp_test_acc = accs
                train_f1, val_f1, tmp_test_f1 = f1s
                val_acc_f1 = (val_acc + val_f1) / 2.
                if val_acc > best_val_acc:
                # if tmp_test_f1 > best_test_f1:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    if epoch> 5:
                        goodAug = True
                    # best_val_acc_f1 = val_acc_f1
                    # best_val_f1 = val_f1
                    # best_test_f1 = tmp_test_f1
                    test_acc = accs[2]
                    test_bacc = baccs[2]
                    test_f1 = f1s[2]
                    # print('hello')
                    CountNotImproved =0
                    print('test_f1 CountNotImproved reset to 0 in epoch', epoch)
                else:
                    goodAug = False
                    CountNotImproved += 1
                end_time = time.time()
                print('epoch: {:3d}, val_loss:{:2f}, acc: {:.2f}, bacc: {:.2f}, tmp_test_f1: {:.2f}, f1: {:.2f}'.format(epoch, val_loss, test_acc * 100, test_bacc * 100, tmp_test_f1*100, test_f1 * 100))
                print(end_time - start_time, file=log_file)
                print(end_time - start_time)
                print('epoch: {:3d}, val_loss:{:2f}, acc: {:.2f}, bacc: {:.2f}, tmp_test_f1: {:.2f}, f1: {:.2f}'.format(epoch, val_loss, test_acc * 100, test_bacc * 100, tmp_test_f1*100, test_f1 * 100),file=log_file)
                end_epoch = epoch
                if CountNotImproved > args.NotImproved:
                    print("No improved for consecutive {:3d} epochs, break.".format(args.NotImproved))
                    break
            if args.IsDirectedData:
                dataset_to_print = args.Direct_dataset.replace('/', '_') + str(args.to_undirected)
            else:
                dataset_to_print = args.undirect_dataset
            print(net_to_print+'layer'+str(args.layer), dataset_to_print,  'EndEpoch', str(end_epoch), 'lr', args.lr)
            print('Split{:3d}, acc: {:.2f}, bacc: {:.2f}, f1: {:.2f}'.format(split, test_acc * 100, test_bacc * 100, test_f1 * 100))
            print(net_to_print, args.layer, dataset_to_print, 'EndEpoch', str(end_epoch), 'lr', args.lr, file=log_file)
            print('Split{:3d}, acc: {:.2f}, bacc: {:.2f}, f1: {:.2f}'.format(split, test_acc * 100, test_bacc * 100, test_f1 * 100), file=log_file)
            macro_F1.append(test_f1*100)
            acc_list.append(test_acc*100)
            bacc_list.append(test_bacc*100)
            if Set_exit:
                sys.exit(1)

        last_time = time.time()
        elapsed_time0 = last_time-start_time
        print("Total time: {} seconds".format(int(elapsed_time0)), file=log_file)
        print("Total time: {} seconds".format(int(elapsed_time0)))
        if len(macro_F1) > 1:
            average = statistics.mean(macro_F1)
            std_dev = statistics.stdev(macro_F1)
            average_acc = statistics.mean(acc_list)
            std_dev_acc = statistics.stdev(acc_list)
            average_bacc = statistics.mean(bacc_list)
            std_dev_bacc = statistics.stdev(bacc_list)
            print(net_to_print+str(args.layer)+'_'+dataset_to_print+ "_acc"+f"{average_acc:.1f}±{std_dev_acc:.1f}"+"_bacc"+f"{average_bacc:.1f}±{std_dev_bacc:.1f}"+'_Macro F1:'+f"{average:.1f}±{std_dev:.1f}")
            print(net_to_print+str(args.layer)+'_'+dataset_to_print+ "_acc"+f"{average_acc:.1f}±{std_dev_acc:.1f}"+"_bacc"+f"{average_bacc:.1f}±{std_dev_bacc:.1f}"+'_Macro F1:'+f"{average:.1f}±{std_dev:.1f}", file=log_file)


except KeyboardInterrupt:
    # If interrupted, the signal handler will be triggered
    pass

