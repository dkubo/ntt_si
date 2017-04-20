#coding:utf-8

import pickle
import collections
import numpy as np

import chainer
from chainer import cuda

import train
import data

def open_model(epoch):
	with open('model'+str(epoch), 'rb') as i:
		model = pickle.load(i)
	return model


if __name__ == '__main__':
	vocab = collections.defaultdict(lambda: len(vocab))

	# データ取得
	train_data, test_data, vocab = data.get_data(vocab)

	# inverse key and value
	inv_vocab = {v:k for k, v in vocab.items()}

	# 文生成
	model = open_model(epoch=100)
	while x == 7:		# 7: EOS
		x = model(np.array(0, dtype=np.int32), ans=None, train=False)
		print x

