#coding:utf-8

class Sentence(object):

	def __init__(self, sentence):
		self.sentence = sentence

class Query(object):		# class: 何も継承しない場合は,objectを指定する

    def __init__(self, sentence, answer, fact):
        self.sentence = sentence
        self.answer = answer
        self.fact = fact

def normalize(sentence):
	return sentence.lower().replace('.', '').replace('?', '')

def get_wid(vocab, words):
	return [vocab[word] for word in words]

def parse_line(vocab, line):
	if '\t' in line:
		#question line
		q, ans, f_sids = line.split('\t')
		ans_id = get_wid(vocab, [ans])[0]	# ans_id: answerの語彙のid
		q_words = normalize(q).split()
		q_ids = get_wid(vocab, q_words)
		f_sids = map(int, f_sids.split(' '))		# map(a,b): aをbの全要素に適用する
		return Query(q_ids, ans_id, f_sids)
	else:
		#sentence line
		s_words = normalize(line).split()
		s_ids = get_wid(vocab, s_words)
		return Sentence(s_ids)


def parse_data(fpath, vocab):
	data = []
	all_data = []
	with open(fpath) as f:
		for line in f:
			# pos: 最初の空白のインデックス
			pos = line.find(' ')	# find(str): strとマッチしたインデックスを返す
			sid = int(line[:pos])	# sid: 文id
			line = line[pos:]			# sidを除去
			if sid == 1 and len(data) > 0:
				all_data.append(data)
				data = []				
			data.append(parse_line(vocab, line))

		if len(data) > 0:
			all_data.append(data)

		return all_data


