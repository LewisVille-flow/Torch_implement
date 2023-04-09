from typing import List, Dict, Tuple, Sequence
import torch
import re

class tokenizer():
    def __self__():
        print("tokenzier setting")

    def re_tokenizer(self, sentence: List[str]) -> List[str]:
        tokens: List[str]
        
        pattern = r'\s+|([.,?!]|n\'t|\'\w+)'
        tokens = list(filter(lambda x:x, re.split(pattern, sentence.lower())))
        
        return tokens
        
    # def byte_pair_tokenzier(self, sentence: List[str]) -> List[str]:
        
if __name__ == '__main__':
    _input = "My work, Jennifer's work, I didn't do it, ma'am. How?"
    test_tok = tokenizer()
    print(test_tok.re_tokenizer(_input))