#coding:utf-8

import collections
import glob
import copy
import argparse
import numpy as np
import data
import cupy

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers, Variable, cuda


"""
python train.py --gpu 0

"""

class MemNN(chainer.Chain):
	def __init__(self, n_vocab, word_embed_size, max_memory=50):
		super(MemNN, self).__init__(
			A_1 = L.EmbedID(n_vocab, word_embed_size, ignore_label=-1),  # encoder for inputs
			A_2 = L.EmbedID(n_vocab, word_embed_size, ignore_label=-1),  # encoder for inputs
			A_3 = L.EmbedID(n_vocab, word_embed_size, ignore_label=-1),  # encoder for inputs
			C_3 = L.EmbedID(n_vocab, word_embed_size, ignore_label=-1),  # encoder for inputs
			W = L.Linear(word_embed_size, n_vocab), 										 # encoder for answer
		)
		# 重みのランダム初期化 (平均0,標準偏差0.1の正規分布)
		init_params(self.A_1, self.A_2, self.A_3, self.C_3, self.W)
		# Adjacent (A_k+1=C_k)
		self.C_1 = self.A_2																							  # encoder for outputs
		self.C_2 = self.A_3 																							 # encoder for outputs
		# Adjacent (B = A_1)
		self.B = self.A_1  																							# encoder for query

	
	def __call__(self, x_input, query, answer, train=True):
		# layer 1
		b_2 = self.memory(x_input, query, layer=1)
		# layer 2
		b_3 = self.memory(x_input, b_2, layer=2)
		# layer 3
		b_4 = self.memory(x_input, b_2, layer=3)

		predict = self.W(b_4)
#		print "answer.shape,predict.shape:", answer.shape,predict.data.shape
		if train:
			return F.softmax_cross_entropy(predict, answer)
		else:
			return F.accuracy(predict, answer)

	# エンコード = 埋め込みベクトルの和
	def encode_input(self, x_input, layer=1):
		if layer == 1:
			return F.sum(self.A_1(x_input), axis=1)
		elif layer == 2:
			return F.sum(self.A_2(x_input), axis=1)
		elif layer == 3:
			return F.sum(self.A_3(x_input), axis=1)

	def encode_query(self, query):
		return F.sum(self.B(query), axis=1)
		
	def encode_output(self, x_input, layer=1):
		if layer == 1:
			return F.sum(self.C_1(x_input), axis=1)
		elif layer == 2:
			return F.sum(self.C_2(x_input), axis=1)
		elif layer == 3:
			return F.sum(self.C_3(x_input), axis=1)

	def memory(self, x_input, query, layer):
		m = self.encode_input(x_input,layer)		# memory for input
		c = self.encode_output(x_input)		# memory for output
		if layer == 1:
			u = self.encode_query(query)			# memory for query
		else:
			u = query
#		print "m.data.shape", m.data.shape		# (50,20)
#		print "u.data.shape", u.data.shape		# (1,20)
		mu = F.matmul(m, u, transb=True)	# mを転置して内積をとる
		p = F.softmax(mu)									# 文の重要度p(アテンション)
#		print p.data.shape		# (50,1)
#		print c.data.shape		# (50,20)
		o = F.matmul(p, c, transa=True)		# cとpのweighted sum
#		print o.data.shape		# (1,20)
		return (u+o)

def init_params(*embs):	# *: 引数をリストとして受け取る
    for emb in embs:
	    emb.W.data[:] = np.random.normal(0, 0.1, emb.W.data.shape)


def convert_data(before_data, gpu):
	d = []	# story: 15 → [[3通常文(mem), 1質問文, 1答え],[6通常文(mem), 1質問文, 1答え],...]
	# 文の最長単語数を求める
	sentence_maxlen = max(max(len(s.sentence) for s in story) for story in before_data)
	for story in before_data:
		mem = np.ones((50, sentence_maxlen), dtype=np.int32)		# mem: 50×sentence_maxlenのint32のゼロ行列
		mem = -1 * mem
		i = 0
		for sent in story:
#			# isinstance(object, class): objectがclassのインスタンスかどうか
			if isinstance(sent, data.Sentence):
				if i == 50:		# The capacity of memory is restricted to the most 50 sentence(1ストーリーあたり50文まで記憶する)
					mem[0:i-1, :] = mem[1:i, :]		# 一番古い情報をシフトする(1〜49→0〜48にシフト)
					# print mem[0,0:3]	# 0行目の0〜2列を取得
#					mem_length[0:i-1] = mem_length[1:i]
					i -= 1
				mem[i, 0:len(sent.sentence)] = sent.sentence
#				mem_length[i] = len(sent.sentence)
				i += 1
			elif isinstance(sent, data.Query):
				# question sentence
				query = np.ones(sentence_maxlen, dtype=np.int32)	# 質問文ベクトル
				query = -1 * query
				query[0:len(sent.sentence)] = sent.sentence
				query = query[np.newaxis, :]		# 1次元→2次元配列に変換
				answer = np.array([sent.answer], dtype=np.int32)
				if gpu >= 0:	# gpu
					d.append((cuda.to_gpu(mem),cuda.to_gpu(query),cuda.to_gpu(answer)))
				else:
					d.append((copy.deepcopy(mem),(query),answer))

	return d


def get_arg():
	parser = argparse.ArgumentParser(description='converter')
	parser.add_argument('-gpu','--gpu', type=int, default=-1)
	args = parser.parse_args()
	return args

def proc(iter_list,train=True):
	total_loss = 0.0
	total_acc = 0.0
	cnt = 0
	for batch_data in iter_list:
#		print len(batch_data)		# 100
		accum_loss = None
#		for b_cnt, batch in enumerate(batch_data):
#		x_input, query, answer = batch
		x_input, query, answer = batch_data[0]
		# train
		if train:
			loss = model(x_input, query, answer, train=True)
			total_loss += loss.data
			if accum_loss is None:
				accum_loss = loss
			else:
				accum_loss += loss
		# test
		else:
			acc = model(x_input, query, answer, train=False)			
			total_acc += acc.data

		cnt += 1
		if accum_loss is not None:
			model.zerograds()
#			optimizer.zero_grads()
			accum_loss.backward()
			optimizer.update()

	if train:
		print "total_loss:", total_loss
	else:	
		ave_acc = float(total_acc) / cnt
		print "ave_acc:", ave_acc

	return float(total_acc) / cnt
	
if __name__ == '__main__':
	args = get_arg()
	print 'gpu:',args.gpu
	root_path = "../../data/tasks_1-20_v1-2/en"
	# 未知語(:k)が引数として与えられた場合、id(:v)を付与する
	vocab = collections.defaultdict(lambda: len(vocab))
	for data_id in range(1,21):
#	data_id = 1
		# glob.glob: マッチしたパスをリストで返す
		fpath = glob.glob('%s/qa%d_*train.txt' % (root_path, data_id))[0]
		train_data = data.parse_data(fpath, vocab)
		fpath = glob.glob('%s/qa%d_*test.txt' % (root_path, data_id))[0]
		test_data = data.parse_data(fpath, vocab)
		print('Training data: %d' % len(train_data))		# 文id=1で区切ったとき(story)のデータ数
		train_data = convert_data(train_data, args.gpu)
		test_data = convert_data(test_data, args.gpu)
		model = MemNN(len(vocab), 20, 50)	# (n_units:word_embeddingの次元数(=20), n_vocab:語彙数, max_mem=50)
		if args.gpu >= 0:
			model.to_gpu()
			xp = cupy
		else:
			xp = np

		# Setup an optimizer	
		optimizer = optimizers.Adam(alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-6)
		optimizer.setup(model)

		batch_size = 1
	#	print len(train_data)		# 1000

		for epoch in range(20):
			print "epoch:", epoch
			train_iter = chainer.iterators.SerialIterator(train_data, batch_size, repeat=False)
			test_iter = chainer.iterators.SerialIterator(test_data, batch_size, repeat=False, shuffle=False)
			proc(train_iter, train=True)
			acc = proc(test_iter, train=False)

		acc = acc * 100			# convert from ratio to %
		err = 100 - acc
		print "acc:", acc



