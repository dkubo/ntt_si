#coding:utf-8

#class PrecedentData():



# 語id取得
def get_wid(vocab, words):
	return [vocab[word] for word in words]


def get_data(vocab):
	train_data = []
	test_data = []
	# 訓練データ
	train_words = ["BOS","私は","太郎","では","なく","四朗","。","EOS"]
	train_data.append(get_wid(vocab, train_words))
	train_words = ["BOS","私は","次郎","では","ありません","。","EOS"]
	train_data.append(get_wid(vocab, train_words))
	train_words = ["BOS","私は","三郎","では","ありません","。","EOS"]
	train_data.append(get_wid(vocab, train_words))
	train_words = ["BOS","私は","五郎","です","。","EOS"]
	train_data.append(get_wid(vocab, train_words))
	# テストデータ
	test_words = ["BOS","あなたは","太郎","では","ありません","。","EOS"]
	test_data.append(get_wid(vocab, test_words))
	test_words = ["BOS","私は","三郎","です","。","EOS"]
	test_data.append(get_wid(vocab, test_words))
	test_words = ["BOS","私は","四郎","です","。","EOS"]
	test_data.append(get_wid(vocab, test_words))
	test_words = ["BOS","私たちは","四郎","と","三郎","です","。","EOS"]
	test_data.append(get_wid(vocab, test_words))
	return train_data, test_data, vocab


