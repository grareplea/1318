import os
import argparse
import sys
import logging
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from utils import sparse_mx_to_torch_sparse_tensor
from dataset import load
import torch.nn.functional as F

def test_evaluator(model, adj, idx_train, idx_test, features, labels, hid_units, nb_classes, xent, conv_adj, cuda_no):
    features = torch.FloatTensor(features).cuda(cuda_no)
    adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj)).cuda(cuda_no)

    embeds = model.embed(features, adj, conv_adj, True, None)

    train_embs = embeds[0, idx_train]

    train_lbls = labels[idx_train]
    accs = []
    wd = 0.01 if dataset == 'citeseer' else 0.0
    # for _ in range(50):
    log = LogReg(hid_units, nb_classes).cuda(cuda_no)
    opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
    # log.cuda()
    for _ in range(300):
        log.train()
        opt.zero_grad()
        logits = log(train_embs)

        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(embeds)
    preds = logits.squeeze(0)
    labels = labels

    acc = torch.sum(preds[idx_test].argmax(1) == labels[idx_test]).float() / idx_test.shape[0]
    accs.append(acc * 100)
    accs = torch.tensor(accs)
    logger.info('evaluation acc is {} with std {}'.format(accs.mean().item(), accs.std().item()))
    return accs.mean().item()

def sim(h1, h2):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())

# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Model_barlow(nn.Module):
    def __init__(self, n_in, n_h):
        super(Model_barlow, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.sigm = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(n_h, affine = False)

    def forward(self, bf1, bf2, ba, bd, adj, num_hop, sparse):
        a_emb = self.gcn1(bf1, ba, sparse)
        b_emb = self.gcn2(bf2, bd, sparse)

        f = lambda x: torch.exp(x)

        n_hop_bemb = b_emb[0]
        for i in range(num_hop):
            n_hop_bemb = adj[0] @ n_hop_bemb

        inter_sim = f(sim(a_emb[0], n_hop_bemb))
        intra_sim = f(sim(a_emb[0], a_emb[0]))
        loss = -torch.log(inter_sim.diag()/
                           (intra_sim.sum(dim=-1) - intra_sim.diag()))

        return loss.mean()

    def embed(self, seq, adj, conv_adj, sparse , msk):
        h_1 = self.gcn1(seq, adj, sparse)

        h_2 = self.gcn1(seq, conv_adj, sparse)

        return (h_1 + h_2).detach()

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


def train(dataset, epochs, patience, lr, l2_coef, hid_dim, sample_size, num_hop, test_conv_num, cuda_no, verbose=True):
    # parameter setting portal
    nb_epochs = epochs
    patience = patience
    batch_size = 1
    lr = lr
    l2_coef = l2_coef
    hid_units = hid_dim
    sample_size = sample_size
    sparse = False

    parameter_dict = {}
    parameter_dict['dataset'] = dataset
    parameter_dict['nb_epochs'] = nb_epochs
    parameter_dict['patience'] = patience
    parameter_dict['lr'] = lr
    parameter_dict['l2_coef'] = l2_coef
    parameter_dict['hid_units'] = hid_units
    parameter_dict['sample_size'] = sample_size
    parameter_dict['n'] = num_hop

    logger.info('parameters: {}'.format(parameter_dict))

    adj, features, labels, idx_train, idx_val, idx_test = load(dataset)

    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    model = Model_barlow(ft_size, hid_units)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    xent = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    cnt_wait = 0
    best = 1e9

    best_acc = 0

    filename = 'graph_power/'

    if os.path.exists(filename + dataset + '_conv_adj_' + str(test_conv_num) + '.pt'):
        conv_adj = torch.load(filename +  dataset + '_conv_adj_' + str(test_conv_num) + '.pt', )
    else:
        conv_adj = adj.copy()
        for i in range(test_conv_num - 1):
            conv_adj = conv_adj @ adj
        conv_adj = torch.FloatTensor(conv_adj)
        torch.save(conv_adj, filename  +  dataset + '_conv_adj_' + str(test_conv_num) + '.pt')

    conv_adj = conv_adj.cuda()

    for epoch in range(nb_epochs):
        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(adj[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        adj_ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        adj_ba = torch.FloatTensor(adj_ba).cuda()

        bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

        bf1 = np.array(bf).reshape(batch_size, sample_size, ft_size)
        bf2 = np.array(bf).reshape(batch_size, sample_size, ft_size)

        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)


        bf1 = torch.FloatTensor(bf1)
        bf2 = torch.FloatTensor(bf2)

        if torch.cuda.is_available():
            bf1 = bf1.cuda()
            bf2 = bf2.cuda()
            ba = ba.cuda()
            bd = bd.cuda()

        model.train()
        optimiser.zero_grad()

        loss = model(bf1, bf2, ba, bd, adj_ba, num_hop, sparse)

        loss.backward()
        optimiser.step()

        logger.info('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if verbose:
            model.eval()
            acc = test_evaluator(model, adj, idx_train, idx_test, features, labels, hid_units, nb_classes,
                                 xent, conv_adj, cuda_no)
            if acc > best_acc:
                best_acc = acc

        if loss < best:
            best = loss
            cnt_wait = 0
            torch.save(model.state_dict(), './checkpoints/{}_{}_{}_{}_{}_{}_{}.pkl'.format(dataset,
                                                                                         patience,
                                                                                         lr,
                                                                                         l2_coef,
                                                                                         hid_dim,
                                                                                         batch_size,
                                                                                         sample_size))
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            if verbose:
                logger.info('Early stopping!')
                logger.info('The best test accuracy is : {}'.format(best_acc))
            break


        if epoch == (nb_epochs - 1):
            logger.info('The best test accuracy is : {}'.format(best_acc))

    return best_acc

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser('GCLGP')
    parser.add_argument('--data', type=str, default='cora', help='Dataset name: cora, citeseer, pubmed')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--epoch', type=int, default=500, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=50, help='Patience')
    parser.add_argument('--lr', type=float, default=0.0001, help='Patience')
    parser.add_argument('--l2_coef', type=float, default=0.0, help='l2 coef')
    parser.add_argument('--n', type=int, default=7, help='n-th Graph Power')
    parser.add_argument('--hidden', type=int, default=4096, help='Hidden dim')
    parser.add_argument('--sample_size', type=int, default=2000, help='Sample size')
    parser.add_argument('--sparse', action='store_true', help='Whether to use sparse tensors')
    parser.add_argument('--cuda', type=int, default=0, help='cuda device')


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)


    torch.cuda.set_device(args.cuda)
    dataset = args.data
    n_runs = args.runs
    epochs = args.epoch
    patience = args.patience
    lr = args.lr
    l2_coef = args.l2_coef
    hid_dim = args.hidden
    sample_size = args.sample_size
    num_hop = args.n
    test_conv_num = args.n

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("./logs/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('./logs/{}.log'.format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # set up checkpoint path
    Path("./checkpoints/").mkdir(parents=True, exist_ok=True)

    logger.info('*******n is' + str(test_conv_num))
    accs = []
    for __ in range(n_runs):
        run_acc = train(dataset,
                        epochs,
                        patience,
                        lr,
                        l2_coef,
                        hid_dim,
                        sample_size,
                        num_hop,
                        test_conv_num,
                        args.cuda)
        accs.append(run_acc)
        logger.info('------n:' + str(test_conv_num) + '-----')
        logger.info('accs are: {}'.format(accs))
        logger.info('Final average acc is {} with std {}'.format(np.mean(accs), np.std(accs)))
        logger.info('test learning rate: ' + str(lr))