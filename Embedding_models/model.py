# 모델은 단순히 Input -> XX layers -> output 변환을 해주면 된다.
import torch.nn as nn
from Embedding_models.constant import EMBED_DIMENSION, EMBED_MAX_NORM

# window -> context 예측: 확률(전체 vocab에 대한 확률 비교(CrossEntropyLoss))
class CBOW(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(
            num_embedding=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )
    
    def forward(self, input):           # input: (B, 2W)
        x = self.embeddings(input)      # (B, 2W, EM_D)
        x = x.mean(axis=1)              # (B, EM_D)
        output = self.linear(x)         # (B, V)
        
        return output


# window sized*2 input -> one context vector
class SkipGram(nn.Module):
    def __init__(self, vocab_size: int):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embeddings(
            num_embedding=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
            
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size
        )
        
    def forward(self, input):           # input: (B, )
        x = self.embeddings(input)      # (B, EM_D)
        output = self.linear(x)         # (B, V)
        
        return output
    

if __name__ == '__main__':

    print("This file is model.py for Embedding_models.")
    print("import test: {EMBED_DIMENSION}, {EMBED_MAX_NORM}")