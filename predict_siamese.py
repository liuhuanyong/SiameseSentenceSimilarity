#!/usr/bin/env python3
# coding: utf-8
# File: siamese_train.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-23

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, Dropout, Lambda, Bidirectional
import os
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SiameseNetwork:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'data/train.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(cur, 'model/tokenvec_bilstm2_siamese_model.h5')
        self.timestamps_file = os.path.join(cur, 'model/timestamps.txt')
        self.word_dict = self.load_worddict()
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 1
        self.BATCH_SIZE = 512
        self.NUM_CLASSES = 20
        self.VOCAB_SIZE = len(self.word_dict)
        self.LIMIT_RATE = 0.95
        self.TIME_STAMPS = self.load_timestamps()
        self.embedding_matrix = self.build_embedding_matrix()
        self.model = self.load_siamese_model()

    '''加载timestamps'''
    def load_timestamps(self):
        timestamps = [i.strip() for i in open(self.timestamps_file) if i.strip()][0]
        return int(timestamps)

    '''加载词典'''
    def load_worddict(self):
        vocabs = [i.replace('\n','') for i in open(self.vocab_path)]
        word_dict = {wd: index for index, wd in enumerate(vocabs)}
        print(len(vocabs))
        return word_dict

    '''对输入的文本进行处理'''
    def represent_sent(self, s):
        wds = [char for char in s if char]
        sent = [[self.word_dict[char] for char in wds]]
        sent_rep = pad_sequences(sent, self.TIME_STAMPS)
        return sent_rep

    '''加载预训练词向量'''
    def load_pretrained_embedding(self):
        embeddings_dict = {}
        with open(self.embedding_file, 'r') as f:
            for line in f:
                values = line.strip().split(' ')
                if len(values) < 300:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict

    '''加载词向量矩阵'''
    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def exponent_neg_manhattan_distance(self, inputX):
        (sent_left, sent_right) = inputX
        return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))

    '''基于欧式距离的字符串相似度计算'''
    def euclidean_distance(self, sent_left, sent_right):
        sum_square = K.sum(K.square(sent_left - sent_right), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))


    '''搭建编码层网络,用于权重共享'''
    def create_base_network(self, input_shape):
        input = Input(shape=input_shape)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input)
        lstm1 = Dropout(0.5)(lstm1)
        lstm2 = Bidirectional(LSTM(32))(lstm1)
        lstm2 = Dropout(0.5)(lstm2)
        return Model(input, lstm2)

    '''搭建网络'''
    def bilstm_siamese_model(self):
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=True)

        left_input = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        right_input = Input(shape=(self.TIME_STAMPS,), dtype='float32')

        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        shared_lstm = self.create_base_network(input_shape=(self.TIME_STAMPS, self.EMBEDDING_DIM))
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        distance = Lambda(self.exponent_neg_manhattan_distance)([left_output, right_output])
        model = Model([left_input, right_input], distance)
        model.compile(loss='binary_crossentropy',
                      optimizer='nadam',
                      metrics=['accuracy'])
        model.summary()
        return model

    '''使用模型'''
    def load_siamese_model(self):
        model = self.bilstm_siamese_model()
        model.load_weights(self.model_path)

        return model

    '''使用模型进行预测'''
    def predict(self, s1, s2):
        rep_s1 = self.represent_sent(s1)
        rep_s2 = self.represent_sent(s2)
        res = self.model.predict([rep_s1, rep_s2])
        return res

    '''测试模型'''
    def test(self):
        s1 = '请问您需要办理什么业务？'
        s2 = '请问您需要办理什么业务？'
        res = self.predict(s1, s2)
        print(res)
        return


if __name__ == '__main__':
    handler = SiameseNetwork()
    handler.test()


