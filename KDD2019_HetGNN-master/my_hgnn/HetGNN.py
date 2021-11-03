import math
import torch
import torch.nn as nn

from data_load import load_en_vocab


class FNN(nn.Module):
    def __init__(self, input_dim, hid_state_dim):
        super(FNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hid_state_dim)
        self.linear2 = nn.Linear(hid_state_dim, input_dim)
        self.activation = nn.ReLU()
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, inputs):
        hid = self.activation(self.linear1(inputs))
        output = self.linear2(hid)

        return self.layernorm(output + inputs)


class HGraph(nn.Module):
    def __init__(self, ):
        super(HGraph, self).__init__()

    def forward(self):
        pass


class PositionEmbedding(nn.Module):
    def __init__(self, input_dim, max_len=50, emb_dropout=0.1):
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.FloatTensor(torch.arange(0., input_dim, 2) * -(math.log(10000.) / input_dim)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.emb_dropout = nn.Dropout(emb_dropout)
        # (1, Max_len, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x(N, L, D)   pe_clip(1, L, D)     batch_first
        pe_clip = self.pe[:, :x.size(1)]
        x = x + pe_clip
        x = self.emb_dropout(x)
        return x


# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model, dropout, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0., max_len).unsqueeze(1)
#         div_term = torch.exp(torch.FloatTensor(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model)))
#
#         pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
#         pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
#         pe = pe.unsqueeze(0)  # [1, max_len, d_model]
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
#         return self.dropout(x)


# 模型的输入直接就是数据 要构建出各个模态的数据并且将其送入HGNN
class My_model(nn.Module):
    def __init__(self, args):
        super(My_model, self).__init__()
        # get data
        # X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, \
        # SRC_emotion, TGT_emotion, Speakers, A = get_data(hp)

        self.dropout_rate = args.dropout_rate  # LSTM需要dropout，HGNN需要dropout
        # utterance
        self.encoder_embedding = nn.Embedding(args.vocab_size, args.word_dim, padding_idx=0,
                                              _weight=args.word_embeddings)
        self.encoder_embedding.weight.requires_grad = False
        self.gru = nn.GRU(args.word_dim, args.rnn_hidden_dim, num_layers=1, dropout=args.dropout_rate)
        self.multihead_attn = nn.MultiheadAttention(args.rnn_hidden_dim, num_heads=8, dropout=self.dropout_rate)


        #load embedding
        embed = self.encoder_embedding()
        h0 = torch.randn(1, args.batch_size, args.rnn_hidden_dim)  # zero hidden state of GRU
        lstm_out, self.gru_hid_state = self.gru(self.encoder_embedding, h0)
        #pe
        position_embedding = PositionEmbedding(args.rnn_hidden_dim)
        self.pe = position_embedding(self.gru_hid_state)
        #这里要拼接一下

        attn_output, attn_output_weights = self.multihead_attn(self.gru_hid_state, self.gru_hid_state,
                                                               self.gru_hid_state)
        # facial vector
        self.facial_expr = FNN(17, args.facial_dim)

        # emotion vector
        self.emotion_embed = nn.Embedding(7, embedding_dim=args.emo_dim)
        # speaker vector
        self.speaker_embed = nn.Embedding(args.spk_num, args.spk_dim)

        # concat vector
        # self.enc_input = torch.concat((self.diag_his, self.facial_expr, self.emotion_embed, self.speaker_embed), dim=0)

        en2idx, idx2en = load_en_vocab(hp)

    def forward(self):
        pass
