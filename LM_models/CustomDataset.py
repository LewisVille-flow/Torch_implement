import torch
from torch.utils.data import DataSet

class CBOWDataset(DataSet):
    def __init__(self, tokenizer):
        # arg 경고
        
        # build vocabulary
        
        # tokenizer setting
        
        # token2id
    
    def __len__(self):
    
    
    def __getitem__(self, idx):
        # token2id 이용해서 인덱싱
        
        # if not in token2id -> return <UNK>
        
        # 텐서 변환
        
        return torch.tensor(x), torch.tensor(y)
    

if __name__ == '__main__':
    
    print("This file is CustomDataset archive.")