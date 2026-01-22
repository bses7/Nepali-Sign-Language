import json
from pathlib import Path

class NSLTokenizer:
    def __init__(self, config=None):
        # Special Tokens
        self.pad_token = "<PAD>" 
        self.sos_token = "<SOS>" 
        self.eos_token = "<EOS>" 
        self.unk_token = "<UNK>"  
        
        self.vocab = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
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

    def tokenize(self, text):
        """Converts a Nepali string into a list of IDs."""
        tokens = [self.char2idx.get(char, self.char2idx[self.unk_token]) for char in text]
        return [self.char2idx[self.sos_token]] + tokens + [self.char2idx[self.eos_token]]

    def save_vocab(self, path="vocab.json"):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char2idx': self.char2idx,
                'idx2char': self.idx2char
            }, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary saved to {path}")

    def load_vocab(self, path="vocab.json"):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char2idx = data['char2idx']

            self.idx2char = {int(k): v for k, v in data['idx2char'].items()}
            self.vocab = list(self.char2idx.keys())
        print(f"📖 Vocabulary loaded from {path}")