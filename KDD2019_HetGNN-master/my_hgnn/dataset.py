# from .utils_function import to_cuda, merge, merge_multi_response, merge_sent_and_word
# from .utils_function import to_cuda, merge, merge_multi_response, merge_sent_and_word
import pickle

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_path):
        super(MyDataset, self).__init__()
        data = pickle.load(open(data_path, 'rb'), encoding='utf-8')
        ' X_image, Y_image, X_length, Y, sources, targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A'
        self.X = data[0]
        self.X_image = data[1]
        self.Y_image = data[2]
        self.Y = data[5]
        self.X_length = data[3]
        self.X_turn_number = data[4]
        self.SRC_emotion = data[5]
        self.TGT_emotion = data[6]
        self.speaker = data[6]
        self.A = data[7]

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        x = torch.LongTensor(self.X[index])
        x_ima = torch.FloatTensor(self.X_image[index])
        y_ima = torch.FloatTensor(self.Y_image[index])
        x_len = torch.LongTensor(self.X_length[index])
        x_turn_num = torch.LongTensor(self.X_turn_number[index])
        y = torch.LongTensor(self.Y[index])
        spe = torch.LongTensor(self.speaker[index])

        return x, x_ima, y_ima, x_len, x_turn_num, y, spe

    def __len__(self):
        return self.X.shape[0]


def collate_fn(data):
    context = []
    context_image = []
    response = []
    response_image = []
    context_lenth = []
    context_turn = []
    speaker = []
    for d in data:
        context.append(d[0])
        context_image.append(d[1])
        response.append(d[-2])
        response_image.append(d[2])
        context_lenth.append(d[3])
        context_turn.append(d[4])
        speaker.append(d[-1])

    context = torch.stack(context, dim=0)
    context_image = torch.stack(context_image, dim=0)
    response = torch.stack(response, dim=0)
    response_image = torch.stack(response_image, dim=0)
    context_lenth = torch.stack(context_lenth, dim=0)
    speaker = torch.stack(speaker, dim=0)

    return context, context_image, response, response_image, context_lenth, speaker

    # def collate_fn_my(self):
    #
    # def load_de_vocab(self, hp):
    #     vocab = [line.split()[0] for line in
    #              codecs.open(self.base_dir + 'preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if
    #              int(line.split()[1]) >= hp.min_cnt]
    #     word2idx = {word: idx for idx, word in enumerate(vocab)}
    #     idx2word = {idx: word for idx, word in enumerate(vocab)}
    #     return word2idx, idx2word
    #
    # def load_en_vocab(self, hp):
    #     vocab = [line.split()[0] for line in
    #              codecs.open(self.base_dir + 'preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if
    #              int(line.split()[1]) >= hp.min_cnt]
    #     word2idx = {word: idx for idx, word in enumerate(vocab)}
    #     idx2word = {idx: word for idx, word in enumerate(vocab)}
    #     return word2idx, idx2word
    #
    # def load_speaker_vocab(self, hp):
    #     vocab = [line.split('\n')[0] for line in
    #              codecs.open(self.base_dir + 'preprocessed/speakers.txt', 'r', 'utf-8').read().splitlines()]
    #     word2idx = {word: idx for idx, word in enumerate(vocab)}
    #     idx2word = {idx: word for idx, word in enumerate(vocab)}
    #     return word2idx, idx2word
    #
    # def create_data(self, hp, source_sents, target_sents, image_fea, A):
    #     de2idx, idx2de = self.load_de_vocab(hp)
    #     en2idx, idx2en = self.load_en_vocab(hp)
    #     speaker2idx, idx2speaker = self.load_speaker_vocab(hp)
    #     # Index
    #
    #     x_A, x_list, x_image_list, y_image_list, y_list, Sources, Targets, Src_emotion, Tgt_emotion, Speaker = [], [], [], [], [], [], [], [], [], []
    #     max_turn = 0
    #     max_length = 0
    #     for index, (source_sent, target_sent, image_f, a) in enumerate(zip(source_sents, target_sents, image_fea, A)):
    #         source_sent_split = source_sent.split(u"</d>")
    #         source_sent_split.pop()
    #         image_feature = image_f.split("\t\t")
    #         image_feature.pop()
    #         image = image_feature[:-1]
    #         y_imag = image_feature[-1]
    #
    #         x = []
    #         x_image = []
    #         y_image = []
    #         src_emotion = []
    #         turn_num = 0
    #         for sss, imag in zip(source_sent_split, image):
    #             if len(sss.split()) == 0 or len(sss.split("\t\t")) == 1:
    #                 print('is 0', index, sss)
    #                 continue
    #             # print(sss)
    #             x_speaker, text, emotion = sss.split("\t\t")[0], sss.split("\t\t")[1], sss.split("\t\t")[2]
    #             if len((text + u" </S>").split()) > max_length:
    #                 max_length = len((text + u" </S>").split())
    #
    #             x.append([de2idx.get(word, 1) for word in (text + u" </S>").split()])  # 1: OOV, </S>: End of Text
    #             x_image.append([float(item) for item in imag.split()])
    #
    #             src_emotion.append([self.emotion2idx[emotion.split()[0]]])
    #             turn_num += 1
    #
    #         target_sent_split = target_sent.split(u"</d>")
    #         if len(x) > max_turn:
    #             max_turn = len(x)
    #
    #         speaker = []
    #         tgt_emotion = []
    #         name = ' '.join(target_sent_split[0].split())
    #         if name not in speaker2idx:
    #             speaker.append(speaker2idx[u"newer"])
    #         else:
    #             speaker.append(speaker2idx[name])
    #         tgt_emotion.append(self.emotion2idx[target_sent_split[2].split()[0]])
    #         src_emotion.append(self.emotion2idx[target_sent_split[2].split()[0]])
    #         y = [en2idx.get(word, 1) for word in (target_sent_split[1] + u" </S>").split()]
    #         y_image.append([float(item) for item in y_imag.split()])
    #
    #         if max(len(x), len(y)) <= hp.maxlen:
    #             x_list.append(np.array(x))
    #             x_image_list.append(np.array(x_image))
    #             y_image_list.append(np.array(y_image))
    #             y_list.append(np.array(y))
    #             Src_emotion.append(np.array(src_emotion))
    #             Tgt_emotion.append(np.array(tgt_emotion))
    #             Speaker.append(np.array(speaker))
    #             Sources.append(source_sent)
    #             Targets.append(target_sent)
    #             x_A.append(a)
    #
    #     # code.interact(local=locals())
    #     print('max_turn=', max_turn)
    #     # Pad
    #     print('max_length=', max_length)
    #     X = np.zeros([len(x_list), hp.max_turn, hp.maxlen], np.int32)
    #     X_image = np.zeros([len(x_list), hp.max_turn, 17], np.float32)
    #     Y_image = np.zeros([len(x_list), 17], np.float32)
    #     Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    #     X_length = np.zeros([len(x_list), hp.max_turn], np.int32)
    #     X_turn_number = np.zeros([len(x_list)], np.int32)
    #     SRC_emotion = np.zeros([len(x_list), hp.max_turn], np.int32)
    #     TGT_emotion = np.zeros([len(y_list)], np.int32)
    #     Speakers = np.zeros([len(y_list)], np.int32)
    #     X_A = np.zeros([len(x_list), 7, 90, 90], np.float32)
    #     for i, (x, y, z) in enumerate(zip(x_list, y_list, x_image_list)):  # i-th dialogue
    #         j = 0
    #         for j in range(len(x)):  # j-th turn
    #             if j >= hp.max_turn:
    #                 break
    #             if len(x[j]) < hp.maxlen:
    #                 X[i][j] = np.lib.pad(x[j], [0, hp.maxlen - len(x[j])], 'constant',
    #                                      constant_values=(0, 0))  # i-th dialogue j-th turn
    #             else:
    #                 X[i][j] = x[j][:hp.maxlen]  #
    #             X_image[i][j] = z[j]
    #             X_length[i][j] = len(x[j])  # seq length mask
    #             SRC_emotion[i][j] = Src_emotion[i][j][0]
    #             # code.interact(local=locals())
    #         X_turn_number[i] = len(x) + 1  # turn number`
    #         Y_image[i] = y_image_list[i]
    #         X_image[i][j + 1] = y_image_list[i]
    #         # code.interact(local=locals())
    #
    #         Y[i] = np.lib.pad(y, [0, hp.maxlen - len(y)], 'constant', constant_values=(0, 0))
    #         TGT_emotion[i] = Tgt_emotion[i][0]
    #         Speakers[i] = Speaker[i][0]
    #         for k in range(len(x_A[i])):
    #             X_A[i][k] = x_A[i][k].toarray()
    #
    #     return X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, X_A
