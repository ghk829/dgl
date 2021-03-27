"""Training GCMC model on the MovieLens data set.

The script loads the full graph to the training device.
"""
import os, time
import argparse
import logging
import random
import string
import numpy as np
import torch as th
import torch.nn as nn
from data import MovieLens, JukeboxDataset
from model import BiDecoder, GCMCLayer
import dgl.function as fn
from utils import get_activation, get_optimizer, torch_total_param_num, torch_net_info, MetricLogger

class DotProduct(nn.Module):
    def __init__(self,src_in_units,dst_in_units):
        super(DotProduct, self).__init__()

        self.Q = nn.Parameter(th.randn(src_in_units, dst_in_units))

    def forward(self, dec_graph, ufeat, ifeat):

        with dec_graph.local_scope():
            dec_graph.nodes['item'].data['h'] = ifeat
            dec_graph.nodes['user'].data['h'] = ufeat
            dec_graph.apply_edges(fn.u_dot_v('h', 'h', 'sr'))
            out = dec_graph.edata['sr']

        return out

    def inference(self, uidfeat, ifeat):

        return uidfeat @ self.Q @ ifeat.T

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self._act = get_activation(args.model_activation)
        self.encoder = GCMCLayer(args.rating_vals,
                                 args.src_in_units,
                                 args.dst_in_units,
                                 args.gcn_agg_units,
                                 args.gcn_out_units,
                                 args.gcn_dropout,
                                 args.gcn_agg_accum,
                                 agg_act=self._act,
                                 share_user_item_param=args.share_param,
                                 device=args.device)

        self.decoder = DotProduct(args.gcn_agg_units,
                                 args.gcn_out_units)

    def forward(self, enc_graph, dec_graph, ufeat, ifeat):
        user_out, item_out = self.encoder(
            enc_graph,
            ufeat,
            ifeat)
        pred_ratings = self.decoder(dec_graph, user_out, item_out)
        return pred_ratings

    def forward2(self, enc_graph, dec_graph, ufeat, ifeat):
        user_out, item_out = self.encoder(
            enc_graph,
            ufeat,
            ifeat)
        pred_ratings = self.decoder(dec_graph, user_out, item_out)
        return pred_ratings, user_out, item_out

    def inference(self, uidfeat, ifeat):
        with th.no_grad():
            return self.decoder.inference(uidfeat, ifeat)

def evaluate(args, net, dataset, segment='valid'):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(args.device)

    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_dec_graph
    else:
        raise NotImplementedError

    # Evaluate RMSE
    net.eval()
    rmse = 1

    return rmse


def evaluate_others(args, net, dataset, segment='valid'):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(args.device)

    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_dec_graph
    else:
        raise NotImplementedError

    # Evaluate RMSE
    net.eval()
    with th.no_grad():
        pred_ratings = net(enc_graph, dec_graph,
                           dataset.user_feature, dataset.movie_feature)
    real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                         nd_possible_rating_values.view(1, -1)).sum(dim=1)

    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    pred_ratings = (real_pred_ratings > 0.5).long().cpu().numpy()
    real_ratings = rating_values.numpy()
    cm = confusion_matrix(real_ratings, pred_ratings)
    print('confusion matrix')
    print(cm)
    precision, recall, fscore, support = precision_recall_fscore_support(real_ratings, pred_ratings)

    return precision, recall, fscore, support

def ndcg(recs, gt):
    import math
    Q, S = 0.0, 0.0
    for u, vs in gt.items():
        rec = recs.get(u, [])
        if not rec:
            continue

        idcg = sum([1.0 / math.log(i + 2, 2) for i in range(len(vs))])
        dcg = 0.0
        for i, r in enumerate(rec):
            if r not in vs:
                continue
            rank = i + 1
            dcg += 1.0 / math.log(rank + 1, 2)
        ndcg = dcg / idcg
        S += ndcg
        Q += 1
    return S / Q

def evaluate_ndcg(args, net, dataset, segment='valid'):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(args.device)

    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_dec_graph
    else:
        raise NotImplementedError

    # Evaluate RMSE
    net.eval()
    with th.no_grad():
        ufeat, ifeat = net.encoder(enc_graph,
                           dataset.user_feature, dataset.movie_feature)
    preds = {}
    gt = {}
    for userid, items in dataset.ref_test_data.items():
        userid = int(userid)
        items = [int(e) for e in items]
        gt[userid] = items
        #pred = (ufeat[dataset.global_user_id_map[userid]] @ ifeat.T).argsort(descending=True)[:100].numpy().tolist()
        pred = net.inference(ufeat[dataset.global_user_id_map[userid]], ifeat).sort(descending=True)[1][:100].cpu().numpy().tolist()
        preds[userid] = dataset.item_map.inverse_transform(pred).tolist()

    return ndcg(preds, gt)

def train(args):
    print(args)
    if args.data_name == 'jukebox':
        dataset = JukeboxDataset('dataset/listen_count.txt')
    else:
        dataset = MovieLens(args.data_name, args.device, use_one_hot_fea=args.use_one_hot_fea, symm=args.gcn_agg_norm_symm,
                        test_ratio=args.data_test_ratio, valid_ratio=args.data_valid_ratio)
    print("Loading data finished ...\n")

    args.src_in_units = dataset.user_feature_shape[1]
    args.dst_in_units = dataset.movie_feature_shape[1]
    args.rating_vals = dataset.possible_rating_values

    ### build the net
    net = Net(args=args)
    net = net.to(args.device)
    nd_possible_rating_values = th.FloatTensor(dataset.possible_rating_values).to(args.device)
    rating_loss_net = nn.MSELoss()
    learning_rate = args.train_lr
    optimizer = get_optimizer(args.train_optimizer)(net.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    ### perpare training data
    train_gt_labels = dataset.train_labels
    train_gt_ratings = dataset.train_truths

    ### prepare the logger
    train_loss_logger = MetricLogger(['iter', 'loss'], ['%d', '%.4f', '%.4f'],
                                     os.path.join(args.save_dir, 'train_loss%d.csv' % args.save_id))
    valid_loss_logger = MetricLogger(['iter', 'ndcg','precision','recall','fscore','support'], ['%d','%.4f', '%.4f','%s','%s','%s','%s'],
                                     os.path.join(args.save_dir, 'valid_loss%d.csv' % args.save_id))
    test_loss_logger = MetricLogger(['iter'], ['%d', '%.4f'],
                                    os.path.join(args.save_dir, 'test_loss%d.csv' % args.save_id))

    ### declare the loss information
    best_valid_rmse = np.inf
    no_better_valid = 0
    best_iter = -1
    count_rmse = 1
    count_num = 1
    count_loss = 0

    dataset.train_enc_graph = dataset.train_enc_graph.int().to(args.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(args.device)
    dataset.valid_enc_graph = dataset.train_enc_graph
    dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(args.device)
    dataset.test_enc_graph = dataset.test_enc_graph.int().to(args.device)
    dataset.test_dec_graph = dataset.test_dec_graph.int().to(args.device)

    print("Start training ...")
    dur = []
    for iter_idx in range(1, args.train_max_iter):
        if iter_idx > 3:
            t0 = time.time()
        net.train()
        unique_item_list = dataset.train['item_id'].unique().tolist()
        batches = []
        count_step = 0
        ufeat, ifeat = net.encoder(dataset.train_enc_graph,
                                   dataset.user_feature, dataset.movie_feature)
        from tqdm import tqdm
        for i, row in tqdm(list(dataset.train.iterrows())):
            user,item,rating = row['user_id'], row['item_id'], row['rating']
            userid = dataset.global_user_id_map[user]
            observed = dataset.train[dataset.train['user_id'] == user]['item_id'].tolist()
            res = set()
            while len(res) < 1:
                sample = random.choice(unique_item_list)
                if sample not in observed:
                    res.add(sample)
                    batches.append((userid, dataset.global_item_id_map[item], dataset.global_item_id_map[sample]))
            if len(batches) == 1024:
                uidfeat = ufeat[[ e[0] for e in batches]]
                posfeat = ifeat[[e[1] for e in batches]]
                negfeat = ifeat[[e[2] for e in batches]]

                pos_scores = uidfeat @ net.decoder.Q @ posfeat.T
                neg_scores = uidfeat @ net.decoder.Q @ negfeat.T

                lmbd = 1e-2
                mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()
                mf_loss = -1 * mf_loss

                regularizer = (th.norm(uidfeat) ** 2 + th.norm(posfeat) ** 2 + th.norm(negfeat) ** 2) / 2
                emb_loss = lmbd * regularizer / uidfeat.shape[0]
                optimizer.zero_grad()
                loss = mf_loss + emb_loss
                count_loss += loss.item()
                loss.backward()
                optimizer.step()
                batches = []
                ufeat, ifeat = net.encoder(dataset.train_enc_graph,
                                           dataset.user_feature, dataset.movie_feature)
                count_step += 1

        if batches:
            uidfeat = ufeat[[e[0] for e in batches]]
            posfeat = ifeat[[e[1] for e in batches]]
            negfeat = ifeat[[e[2] for e in batches]]

            pos_scores = uidfeat @ posfeat.T
            neg_scores = uidfeat @ negfeat.T

            lmbd = 1e-2
            mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()
            mf_loss = -1 * mf_loss

            regularizer = (th.norm(uidfeat) ** 2 + th.norm(posfeat) ** 2 + th.norm(negfeat) ** 2) / 2
            emb_loss = lmbd * regularizer / uidfeat.shape[0]
            loss = mf_loss + emb_loss
            count_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)
            optimizer.step()
            batches = []
            ufeat, ifeat = net.encoder(dataset.train_enc_graph,
                                       dataset.user_feature, dataset.movie_feature)
            count_step += 1

        if iter_idx > 3:
            dur.append(time.time() - t0)

        if iter_idx == 1:
            print("Total #Param of net: %d" % (torch_total_param_num(net)))
            print(torch_net_info(net, save_path=os.path.join(args.save_dir, 'net%d.txt' % args.save_id)))

        if iter_idx % args.train_log_interval == 0:
            train_loss_logger.log(iter=iter_idx,
                                  loss=count_loss / (count_step + 1))
            logging_str = "Iter={}, loss={:.4f}, rmse={:.4f}, time={:.4f}".format(
                iter_idx, count_loss/(count_step + 1), count_rmse/count_num,
                np.average(dur))
            count_rmse = 1
            count_num = 1
            count_step = 0

        if iter_idx % args.train_valid_interval == 0:
            valid_rmse = evaluate(args=args, net=net, dataset=dataset, segment='valid')
            precision, recall, fscore, support = evaluate_others(args=args, net=net, dataset=dataset, segment='valid')
            ndcg = evaluate_ndcg(args=args, net=net, dataset=dataset, segment='valid')
            print('ndcg', ndcg, 'precision', precision, 'recall', recall, 'fscore', fscore, 'support', support)
            valid_loss_logger.log(iter=iter_idx, ndcg=ndcg, precision=precision, recall=recall, fscore=fscore,
                                  support=support)
            logging_str += ',\tVal RMSE={:.4f}'.format(valid_rmse)

            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                no_better_valid = 0
                best_iter = iter_idx
                test_rmse = evaluate(args=args, net=net, dataset=dataset, segment='test')
                best_test_rmse = test_rmse
                test_loss_logger.log(iter=iter_idx)
                logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
            else:
                no_better_valid += 1
                if no_better_valid > args.train_early_stopping_patience\
                    and learning_rate <= args.train_min_lr:
                    logging.info("Early stopping threshold reached. Stop training.")
                    break
                if no_better_valid > args.train_decay_patience:
                    new_lr = max(learning_rate * args.train_lr_decay_factor, args.train_min_lr)
                    if new_lr < learning_rate:
                        learning_rate = new_lr
                        logging.info("\tChange the LR to %g" % new_lr)
                        for p in optimizer.param_groups:
                            p['lr'] = learning_rate
                        no_better_valid = 0
        if iter_idx  % args.train_log_interval == 0:
            print(logging_str)
    print('Best Iter Idx={}, Best Valid RMSE={:.4f}, Best Test RMSE={:.4f}'.format(
        best_iter, best_valid_rmse, best_test_rmse))
    train_loss_logger.close()
    valid_loss_logger.close()
    test_loss_logger.close()


def config():
    parser = argparse.ArgumentParser(description='GCMC')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--data_name', default='ml-1m', type=str,
                        help='The dataset name: ml-100k, ml-1m, ml-10m')
    parser.add_argument('--data_test_ratio', type=float, default=0.1) ## for ml-100k the test ration is 0.2
    parser.add_argument('--data_valid_ratio', type=float, default=0.1)
    parser.add_argument('--use_one_hot_fea', action='store_true', default=False)
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--gcn_dropout', type=float, default=0.3)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_units', type=int, default=10)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=10)
    parser.add_argument('--gen_r_num_basis_func', type=int, default=1)
    parser.add_argument('--train_max_iter', type=int, default=2000)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=50)
    parser.add_argument('--train_early_stopping_patience', type=int, default=100)
    parser.add_argument('--share_param', default=True, action='store_true')

    args = parser.parse_args()
    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')

    ### configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == '__main__':
    args = config()
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)
    train(args)
