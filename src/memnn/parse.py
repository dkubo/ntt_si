#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

def file_list():
    basename = '../../data/tasks_1-20_v1-2/en/'
    files = os.listdir(basename)
    files = sorted([f for f in files if '.txt' in f])
    files = [(int(f.split('_')[0].replace('qa', '')) ,f) for f in files]	#:q_id:qa1→1
    train_list = []
    test_list = []
    #trainとtestに分割する(ファイル名から)
    for q_id, filename in files:
        if '_train' in filename:
            train_list.append((q_id, basename + filename))
        else:
            test_list.append((q_id, basename + filename))
    train_filenames = dict(train_list)		#k:q_id,fname
    test_filenames = dict(test_list)			#k:q_id,fname
    return train_filenames, test_filenames


def r(task_id, filename, vocab):		#filename:ファイルへの相対パス
    batch_list = []
    item_fact = []
    item_qa = []
    lines = [l for l in open(filename)]					#filenameを開いて、一行ごと配列に格納
    for i, l in enumerate(lines):								#インデックス(i=0,1,...)付きでループ
        lsplit = l.strip().split('\t')					#質問文にくっついている情報を分離
        text = lsplit[0].replace('?', ' ?').replace('!', ' !').replace('.', ' .').lower()	#文の末尾の記号正規化
        s_id = int(text.split()[0])							#文id取得→数値型に変換
        sentence_split = text.split()[1:]				#文と付随する情報を空白で分かち書きした配列
        sentence_str = ' '.join(sentence_split)	#文を空白で分かち書きした文字列
				#文セットの初め(s_id=1)かつそれが一番初めの行でない(i!=0) or 文末(i == len(lines))
        if (s_id == 1 and i != 0) or i == len(lines):			#lines:1ファイル内から読み込んだ一文ごとの配列
#            print '***'
#            print item_fact
#            print item_qa
            batch_list.append((item_fact, item_qa))
            item_fact = []
            item_qa = []
        if len(lsplit) == 1:		#質問文ではない普通の文のとき
            sentence_idx = build_vocab(vocab, sentence_split)
            item_fact.append((task_id, s_id, sentence_idx, sentence_str))
        else:			#質問文のとき
            # answer = lsplit[1].lower().split(',')
            answer = [lsplit[1].lower()]
#            print "answer:"+answer[0]
            answer_hint = map(int, lsplit[2].split())
#            print answer_hint
            sentence_idx = build_vocab(vocab, sentence_split)	#vocab:Hash
            answer_idx = build_vocab(vocab, answer)
            item_qa.append((task_id, s_id, sentence_idx, sentence_str, answer_idx, answer, answer_hint))
        # print s_id, 
        # print sentence_str

        # print lsplit

    # print batch_list[-1]
    return batch_list

def build_vocab(vocab, word_list):
    index_list = []
    for w in word_list:
        if w not in vocab:
            vocab[w] = len(vocab)
        index_list.append(vocab[w])
    return index_list


def build_dataset():
    train_filenames, test_filenames = file_list()
    vocab = {}
    train_dataset = []
    test_dataset = []
    for task_id in range(1, 20 + 1):		#1〜20
        train_filename = train_filenames[task_id]
        train_batch_list = r(task_id, train_filename, vocab)
        for fact, qa in train_batch_list:
            train_dataset.append((fact, qa))
        test_filename = test_filenames[task_id]
        test_batch_list = r(task_id, test_filename, vocab)
        for fact, qa in test_batch_list:
            test_dataset.append((fact, qa))
        # print train_batch_list
        # train_dataset.append(train_batch_list)
        # test_dataset.append(test_batch_list)
    #語彙の数確認
#    print len(set([i for i, _ in vocab.items()]))		#_:残りのやつらが全部格納される,set(array):重複のない行列
#    print len(set([i.lower() for i, _ in vocab.items()]))		#items:キーと値のリストを取得
    return train_dataset, test_dataset, vocab


if __name__ == '__main__':
    build_dataset()
