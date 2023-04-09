from typing import List, Dict, Tuple, Sequence
from collections import Counter, defaultdict
from itertools import chain
import torch
import re

MAX_VOCAB_SIZE = 30000
END_TOKEN='##'

class Vocabulary():
    special_tokens = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    
    def __init__(self, sentences: List[str], min_freq=1, tokenizer=None):
        #self.tokenizer = split
        if tokenizer:
            self.tokenizer=tokenizer
            self.sentences: List[List[str]] = [self.tokenizer(sentence) for sentence in sentences]
        else:
            # tokenizer = basic splitter
            self.tokenizer=None
            self.sentences: List[List[str]] = [sentence.split() for sentence in sentences]
        
        self.token2id: Dict[str, int] = None
        self.id2token: List[str] = None
        
        self.basic_build_vocab(min_freq=min_freq)
            
    def basic_build_vocab(self, min_freq):
        self.special_tokens_list = list(self.special_tokens.keys())
        
        self.id2token = self.special_tokens_list + [token for token, count in Counter(chain(*self.sentences)).items() if count >= min_freq]
        self.token2id = {token:idx for idx, token in enumerate(self.id2token)}
    
    def sentence_to_numeric(self, text: str) -> List[str]:
        # need to return UNK if there is no such word
        # tokenizer = basic splitter
        if self.tokenizer:
            tokens = self.tokenizer(text)
        else:
            tokens = text.split()
        
        token_ids = [self.token2id[token] if token in self.token2id else self.special_tokens['<UNK>'] for token in tokens]
        
        return token_ids 
        
    def numeric_to_sentence(self, ids: list) -> str:
        return ' '.join(self.id2token[id] for id in ids)


class Byte_Pair_Voucabulary():
    def __init__(self, sentences: List[str], min_freq=1, tokenizer=None):
        #self.tokenizer = split
        if tokenizer:
            self.tokenizer=tokenizer
            self.sentences: List[List[str]] = [self.tokenizer(sentence) for sentence in sentences]
        else:
            # tokenizer = basic splitter
            self.tokenizer=None
            self.sentences: List[List[str]] = [sentence.split() for sentence in sentences]
        
        self.token2id: Dict[str, int] = None
        self.id2token: List[str] = None
        
        self.byte_pair_build_vocab(min_freq=min_freq)
    
    # need to be optimized more!!
    def byte_pair_build_vocab(self, min_freq, max_vocab_size=MAX_VOCAB_SIZE) -> List[str]:
        #self.special_tokens_list = list(self.special_tokens.keys())
        
        id2token = []
        id2token_set = set()
        # 1. 등장하는 단어의 개수를 counter 처리 해서 딕셔너리로 만듦
        _count = dict(Counter(chain(*self.sentences)))
        vocab = defaultdict(int)
        
        # 2. 해당 단어들을 알파벳 크기로 분할한 딕셔너리(vocab) 제작
        _set = set()
        for key, item in _count.items():
            new_key = " ".join(key)
            _split = new_key.split()

            vocab[new_key+" " + END_TOKEN+ " "] = item          # vocab
            id2token_set.update(_split)             # dict ->  중복이 없어야 함....
        id2token_set.add(END_TOKEN)
        #print(id2token)
        
        # 모든 단어가 추가된다는 조건은 ?    
        while len(id2token_set) < max_vocab_size:
            # 3. 거기서 가장 많이 등장하는 pair 확인
            pair_counts = defaultdict(int)

            for token, count in vocab.items():
                _split = token.split()
                for i in range(len(_split)-1):
                    pair_counts[_split[i], _split[i+1]] += count
            #print("pair counts: ", pair_counts)

            # 4. 딕셔너리에 그 max pair 추가
            if not pair_counts:
                break
            max_pair = max(pair_counts, key=pair_counts.get)
            id2token_set.add(''.join(max_pair))
            #print("max pair: ", max_pair)

            updated_vocab = {}
            pattern = re.escape(' '.join(max_pair))
            #p = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')
            p = re.compile(pattern)
            for token, count in vocab.items():
                new_key = p.sub(''.join(max_pair), token)
                updated_vocab[new_key] = count
            #print("new vocab: ", updated_vocab)
            vocab = updated_vocab
            #print("dict: ", id2token_set)
        # 5. vocab 크기에 도달할 때까지
        id2token = id2token_set
        id2token = sorted(id2token, key=len, reverse=True)

        print("id2token: \n", id2token)
    
        self.id2token = list(id2token)
    
    ## OOV 기능 추가해야함!
    def sentence_to_numeric(self, text: str) -> List[str]:
        token_ids: List[int] = None

        end_pattern = r'\s'
        sentence = re.sub(end_pattern, END_TOKEN, text)
        sentence += END_TOKEN
        # 1. 가장 긴 토큰과 매칭되는 지 확인
        # 해당 위치의 값을 substitute
        for idx, vo in enumerate(self.id2token):
            if sentence.find(vo) != -1:
                #print("vo: ", vo, str(idx)+" ")
                sentence = re.sub(vo, str(idx)+" ", sentence)
        #sub = [sentence.find(vo)+1 for idx, vo in enumerate(id2token) if sentence.find(vo) != -1]
        print(sentence)
        token_ids = list(map(int, sentence.split()))
        print(token_ids)
        return token_ids
        
    def numeric_to_sentence(self, ids: List[str]):
        sentence: str = None

        # 1. 변환 후 합치기
        sentence = "".join(self.id2token[id] for id in ids)
        # 2. WORDEND 제거
        end_pattern = re.compile(END_TOKEN)
        sentence = sentence[:-1]
        # 3. _ -> \s
        sentence = end_pattern.sub(" ", sentence)
        ### END YOUR CODE
        return sentence
    
    
if __name__ == '__main__':
    _list = ['hello', 'hi, nice to meet you.', 'great again']
    test_vocab = Vocabulary(_list, min_freq=1)
    
    basic_test_numeric = test_vocab.sentence_to_numeric('hello, nice to meet you')
    basic_test_tokens = test_vocab.numeric_to_sentence(basic_test_numeric)
    print("test sentence: ", 'hello, nice to meet you')
    print(basic_test_tokens)
    
    bpe_test_vocab = Byte_Pair_Voucabulary(_list, min_freq=1)
    
    bpe_test_numeric = bpe_test_vocab.sentence_to_numeric('hiello, great to meet you,')
    bpe_test_tokens = bpe_test_vocab.numeric_to_sentence(bpe_test_numeric)
    
    print(bpe_test_tokens)