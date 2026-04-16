import json
from pathlib import Path

class NSLTokenizer:
    def __init__(self, config=None):
        self.pad_token = "<PAD>" 
        self.sos_token = "<SOS>" 
        self.eos_token = "<EOS>" 
        self.unk_token = "<UNK>"
        
        self.sign_mode = "<SIGN>"  
        self.trans_mode = "<TRANS>"
        
        self.vocab = [
            self.pad_token, self.sos_token, self.eos_token, self.unk_token,
            self.sign_mode, self.trans_mode
        ]
        self.char2idx = {}
        self.idx2char = {}
        
        if config:
            self.build_vocab(config)

    def build_vocab(self, config):
        vowels = list(config['processing']['label_map'].values())
        consonants = list(config['processing']['consonant_label_map'].values())
        
        unique_chars = sorted(list(set(vowels + consonants)))
        
        self.vocab.extend(unique_chars)
        
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        print(f"Vocabulary built! Size: {len(self.vocab)} tokens.")
        print(f"Control Tokens: {self.vocab[:6]}")

    def tokenize(self, text, mode="sign"):
        """
        Updated tokenize method to handle source-target context for transitions.
        Args:
            text: If mode="sign", a single character (e.g. "क").
                  If mode="trans", the characters to/from (e.g. "कख" or ["क", "ख"]).
            mode: "sign" or "trans"
        """
        token_ids = [self.char2idx[self.sos_token]]
        
        if mode == "sign":
            # Structure: [SOS, <SIGN>, character, EOS]
            token_ids.append(self.char2idx[self.sign_mode])
            char = text[0] if len(text) > 0 else self.unk_token
            token_ids.append(self.char2idx.get(char, self.char2idx[self.unk_token]))
        
        else:
            # Structure: [SOS, char_from, <TRANS>, char_to, EOS]
            if len(text) >= 2:
                char_from = text[0]
                char_to = text[1]
            elif len(text) == 1:
                char_from = self.unk_token
                char_to = text[0]
            else:
                char_from = self.unk_token
                char_to = self.unk_token

            token_ids.append(self.char2idx.get(char_from, self.char2idx[self.unk_token]))
            token_ids.append(self.char2idx[self.trans_mode])
            token_ids.append(self.char2idx.get(char_to, self.char2idx[self.unk_token]))
            
        token_ids.append(self.char2idx[self.eos_token])
        return token_ids

    def save_vocab(self, path="vocab.json"):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char2idx': self.char2idx,
                'idx2char': self.idx2char,
                'vocab': self.vocab
            }, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary saved to {path}")

    def load_vocab(self, path="vocab.json"):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char2idx = data['char2idx']
            self.idx2char = {int(k): v for k, v in data['idx2char'].items()}
            self.vocab = data.get('vocab', list(self.char2idx.keys()))
        print(f"Vocabulary loaded. Total tokens: {len(self.vocab)}")