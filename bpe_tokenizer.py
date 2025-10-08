import json
from collections import defaultdict

class BpeTokenizer:
    """
    A Byte-Pair Encoding (BPE) Tokenizer.
    This is a byte-level BPE tokenizer that learns merges from a corpus and can
    encode/decode text based on UTF-8 bytes.
    """
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        # Pre-assign special tokens to fixed IDs
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}

    def _get_stats(self, ids):
        """Counts the frequency of all adjacent pairs of integers in a list of lists."""
        counts = defaultdict(int)
        for sublist in ids:
            for pair in zip(sublist, sublist[1:]):
                counts[pair] += 1
        return counts

    def _merge(self, ids, pair, idx):
        """Replaces all occurrences of a pair in a list of lists with a new integer."""
        new_ids = []
        for sublist in ids:
            new_sublist = []
            i = 0
            while i < len(sublist):
                if i < len(sublist) - 1 and (sublist[i], sublist[i+1]) == pair:
                    new_sublist.append(idx)
                    i += 2
                else:
                    new_sublist.append(sublist[i])
                    i += 1
            new_ids.append(new_sublist)
        return new_ids

    def train(self, corpus: list, vocab_size: int):
        """
        Trains the tokenizer on a given corpus.
        
        Args:
            corpus (list[str]): A list of sentences to train on.
            vocab_size (int): The target size of the vocabulary.
        """
        if vocab_size < 256 + len(self.special_tokens):
            raise ValueError("Vocab size must be at least 256 + number of special tokens")

        num_merges = vocab_size - 256 - len(self.special_tokens)
        text_bytes = [s.encode("utf-8") for s in corpus]
        ids = [list(b) for b in text_bytes]

        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats: break
            best_pair = max(stats, key=stats.get)
            new_id = 256 + len(self.special_tokens) + i
            ids = self._merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id

    def get_vocab_size(self):
        vocab = {token: i for i, token in enumerate(self.special_tokens)}
        for i in range(256):
            vocab[chr(i)] = i + len(self.special_tokens)
        temp_merges_vocab = {}
        for (p0, p1), idx in self.merges.items():
            s0 = temp_merges_vocab.get(p0, chr(p0))
            s1 = temp_merges_vocab.get(p1, chr(p1))
            temp_merges_vocab[idx] = s0 + s1
            vocab[s0 + s1] = idx + len(self.special_tokens)
        self.vocab = {k:v for k,v in sorted(vocab.items(), key=lambda item: item[1])}
        return len(self.vocab)

    def encode(self, text: str) -> list:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = defaultdict(int)
            for pair in zip(ids, ids[1:]): stats[pair] += 1
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges: break
            idx = self.merges[pair]
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                    new_ids.append(idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def decode(self, token_ids):
        byte_list = []
        for token_id in token_ids:
            if token_id in self.special_tokens_map_inv:
                 byte_list.extend(self.special_tokens_map_inv[token_id].encode('utf-8'))
                 continue
            def get_bytes(tid):
                if tid < 256: return [tid]
                pair = next((p for p, i in self.merges.items() if i == tid), None)
                if pair is None: return [self.vocab.get('<unk>', 3)]
                return get_bytes(pair[0]) + get_bytes(pair[1])
            byte_list.extend(get_bytes(token_id))
        try:
            text = bytes(byte_list).decode("utf-8", errors="replace")
        except:
            text = "<decoding error>"
        return text

    def save(self, filepath):
        tokenizer_state = {
            "vocab": self.vocab,
            "merges": {f"{p[0]},{p[1]}": v for p, v in self.merges.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_state, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_state = json.load(f)
        self.vocab = tokenizer_state["vocab"]
        self.merges = {tuple(map(int, k.split(','))): v for k, v in tokenizer_state["merges"].items()}
        self.special_tokens_map_inv = {i: s for s, i in self.vocab.items() if s in self.special_tokens}
