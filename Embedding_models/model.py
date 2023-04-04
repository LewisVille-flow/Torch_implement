# 모델은 단순히 Input -> XX layers -> output 변환을 해주면 된다.
import torch.nn as nn

# window -> context 예측: 확률(전체 vocab에 대한 확률 비교(CrossEntropyLoss))
class CBOW(nn.Module):
    def __init__(self, vocab_size: int):
        # x: (window size*2, 1), y: (context word, 1)
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
    
    def forward(self, input):
        x = self.embeddings(input)
        x = x.mean(axis=1)
        output = self.linear(x)
        
        return output


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embeddings(
            
            
        )
        self.linear = nn.Linear(
            
            out_features=vocab_size
        )
        
    def forward(self, input):
        # input = (batch, seq_len, 1) = (batch, 1, 1)
        x = self.embeddings(input)
        output = self.linear(x)
        
        return output