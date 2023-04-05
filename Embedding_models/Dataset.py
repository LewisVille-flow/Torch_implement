import torch
from torch.utils.data import DataSet
from Embedding_models.constant import CBOW_WINDOW_SIZE, SKIPGRAM_WINDOW_SIZE


# 토큰화 안 되서 들어온다고 가정, 먼저 train 데이터 한정.
class CBOWDataset(DataSet):
    def __init__(self, sentence):
        # arg 경고
        # tokenizer setting
        #self.tokenizer
        
        # build vocabulary
        
        # token2id
        self.token2id
        self.id2token
        
    def __len__(self):
    
    
    def __getitem__(self, idx):
        # token2id 이용해서 인덱싱
        
        # if not in token2id -> return <UNK>
        
        # 텐서 변환
        
        return torch.tensor(x), torch.tensor(y)
    

if __name__ == '__main__':
    
    print("This file is CustomDataset archive.")