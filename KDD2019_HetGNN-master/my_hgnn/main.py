# import torch
# import torch.nn as nn
#
# from HetGNN import My_model,FNN
# from hyperparams import Hyperparams as hp
import argparse
import os

from torch.utils.data import DataLoader

from data_load import load_en_vocab
from dataset import MyDataset, collate_fn

parser = argparse.ArgumentParser(description='Translate script')
base_dir = os.getcwd() + '/'
parser.add_argument('--source_train', type=str, default=base_dir + 'corpora/train_query.txt', help='src train file')
parser.add_argument('--target_train', type=str, default=base_dir + 'corpora/train_answer.txt', help='src train file')
parser.add_argument('--source_test', type=str, default=base_dir + 'corpora/test_query.txt', help='src test file')
parser.add_argument('--target_test', type=str, default=base_dir + 'corpora/test_answer.txt', help='tgt test file')
parser.add_argument('--source_dev', type=str, default=base_dir + 'corpora/dev_query.txt', help='src dev file')
parser.add_argument('--target_dev', type=str, default=base_dir + 'corpora/dev_answer.txt', help='tgt dev file')
parser.add_argument('--corpora_path', type=str, default=base_dir + 'corpora/', help='image file')
parser.add_argument('--logdir', type=str, default='logdir2020_test', help='logdir')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--hidden_units', type=int, default=512, help='context encoder hidden size')
parser.add_argument('--num_blocks', type=int, default=6, help='num_blocks')
parser.add_argument('--num_heads', type=int, default=8, help='num_heads')
parser.add_argument('--maxlen', type=int, default=50, help='maxlen')
parser.add_argument('--min_cnt', type=int, default=1, help='min_cnt')
parser.add_argument('--num_epochs', type=int, default=20000, help='num_epochs')
parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
parser.add_argument('--max_turn', type=int, default=35, help='max_turn')
parser.add_argument('--sinusoid', dest='sinusoid', action='store_true')
hp = parser.parse_args()
print('[!] Parameters:')
print(hp)
# Load vocabulary
# de2idx, idx2de = load_de_vocab(hp)
en2idx, idx2en = load_en_vocab(hp)

# Construct graph
# g = Graph(hp, "train");
print("Graph loaded")

trn_dataset = MyDataset("trn.pkl", )
dev_dataset = MyDataset("dev.pkl")
tst_dataset = MyDataset("tst.pkl")
trn_loader = DataLoader(dataset=trn_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn)
tst_loader = DataLoader(dataset=tst_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn)
