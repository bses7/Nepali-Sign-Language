import json
from pathlib import Path

class NSLTokenizer:
    def __init__(self, config=None):
        # Special Control Tokens
        self.pad_token = "<PAD>" 
        self.sos_token = "<SOS>" 
        self.eos_token = "<EOS>" 
        self.unk_token = "<UNK>"
        
        # Mode Tokens: These tell the model WHAT to do with the character
        self.sign_mode = "<SIGN>"   # "Stay in this pose"
        self.trans_mode = "<TRANS>" # "Move towards this pose"
        
        self.vocab = [
            self.pad_token, self.sos_token, self.eos_token, self.unk_token,
            self.sign_mode, self.trans_mode
        ]
        self.char2idx = {}
        self.idx2char = {}
        
        if config:
            self.build_vocab(config)

    def build_vocab(self, config):
        # Extract vowels and consonants from config maps
        vowels = list(config['processing']['label_map'].values())
        consonants = list(config['processing']['consonant_label_map'].values())
        
        # Ensure we have a unique list of Nepali characters
        unique_chars = sorted(list(set(vowels + consonants)))
        
        # Add characters to vocabulary
        self.vocab.extend(unique_chars)
        
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        print(f"Vocabulary built! Size: {len(self.vocab)} tokens.")
        print(f"Control Tokens: {self.vocab[:6]}")

    def tokenize(self, text, mode="sign"):
        """
        Converts a Nepali string into a list of IDs, prefixed by a mode.
        Args:
            text: The Nepali character(s)
            mode: "sign" for static pose, "trans" for movement
        """
        mode_token = self.sign_mode if mode == "sign" else self.trans_mode
        
        # Start with SOS and the Mode
        token_ids = [self.char2idx[self.sos_token], self.char2idx[mode_token]]
        
        # Add the actual character tokens
        for char in text:
            token_ids.append(self.char2idx.get(char, self.char2idx[self.unk_token]))
            
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
            # JSON keys are always strings, convert back to int for idx2char
            self.idx2char = {int(k): v for k, v in data['idx2char'].items()}
            self.vocab = data.get('vocab', list(self.char2idx.keys()))
        print(f"📖 Vocabulary loaded. Total tokens: {len(self.vocab)}")