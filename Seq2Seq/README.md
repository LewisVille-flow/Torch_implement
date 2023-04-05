## Sequence to Sequence with Attention

__Modules__
1. Encoder module
2. Attention module
    - Dot product(general)
    - Concat attention(a.k.a Bahdanau attention)
3. Decoder module

### Module Explain

1. Encoder: RNN hidden 계산
    - Pack padded sequence
    - Bi-RNN, Hidden concat
    - concatenated hidden - linear(사이즈 줄이기)  
      
2. Attention(dot product): 디코더에서 매번 호출됨.
    - Decoder prev_hidden, Encoder hidden(j) 를 query, key 로 취급
    - Decoder prev_hidden * Encoder Hidden(j) = Energy e
    - weighted sum of (Encoder Hidden * Attention score): j=1~Xt
    
3. Decoder: 어텐션 모듈 호출후 가중치 계산, 다음단어 확률 출력
    - 이전 예측 단어 prev_y RNN embedding
    - RNN Hidden state 계산(prev_y, prev_h)
    - now_h, encoder_h 으로 어텐션 모듈 호출
    - [rnn_output ; attn_value] to linear


4. (Extra) Bahdanau Attention: dot보다 약간 복잡함 
    - e = tanh(W(디코더;인코더)) : 이게 제일 다른 점
    - 에너지 바탕으로 attn_score, attn_value
    
5. (Extra) Bahdanau 에 걸맞는 디코더: 순서를 달리 해줘야 함.
    - 임베딩 -> 바로 어텐션 계산 -> 그러고 나서 RNN Hidden_state 계산(Hidden_state 계산 순서가 다르다)