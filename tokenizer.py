from tiktoken._educational import SimpleBytePairEncoding
import pickle

gpt2_pattern = (
    r"""'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BaseTokenizer:
    def __init__(self):
        self._vocab_size = 0

    def train(self, text, vocab_size):
        raise NotImplemented()

    def encode(self, text):
        raise NotImplemented()

    def decode(self, ints):
        raise NotImplemented()

    @property
    def vocab_size(self):
        return self._vocab_size


class TikTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self._bpe = None

    def train(self, text, vocab_size):
        self._vocab_size = vocab_size
        self._bpe = SimpleBytePairEncoding.train(text, vocab_size, gpt2_pattern)

    def encode(self, text):
        return self._bpe.encode(text, None)

    def decode(self, ints):
        return self._bpe.decode(ints)


class IndexPairTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self._stats = {}
        self._stoi = {}
        self._itos = {}

    def train(self, text, vocab_size):
        chars = sorted(list(set(text)))
        len_chars = len(chars)
        vocab_size = min(len_chars * (len_chars + 1) // 2, vocab_size)
        self._vocab_size = vocab_size
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = {i: ch for i, ch in enumerate(chars)}
        ints = [self._stoi[ch] for ch in text]
        iters = vocab_size - len_chars
        for i in range(iters):
            new_int = len_chars + i
            stats = self._get_stats(ints)
            pair = max(stats, key=stats.get)
            ints = self._merge(ints, pair, new_int)
            self._stoi[pair] = new_int
            self._itos[new_int] = self._itos[pair[0]] + self._itos[pair[1]]

    def encode(self, text):
        ints = [self._stoi[ch] for ch in text]
        while len(ints) >= 2:
            stats = self._get_stats(ints)
            pair = min(stats, key=lambda p: self._stoi.get(p, float("inf")))
            if pair not in self._stoi:
                break
            i = self._stoi[pair]
            ints = self._merge(ints, pair, i)
        return ints

    def decode(self, ints):
        return "".join([self._itos[i] for i in ints])

    @staticmethod
    def _get_stats(ints):
        counts = {}
        for pair in zip(ints, ints[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(ints, pair, new_int):
        new_ints = []
        i = 0
        while i < len(ints):
            if ints[i] == pair[0] and i + 1 < len(ints) and ints[i + 1] == pair[1]:
                new_ints.append(new_int)
                i += 2
            else:
                new_ints.append(ints[i])
                i += 1
        return new_ints


def train_tokenizer(vocab_size, input, output):
    i = open(input, "r")
    o = open(output, "wb")
    bpe = IndexPairTokenizer()
    bpe.train(i.read(), vocab_size)  # , gpt2_pattern)
    pickle.dump(bpe, o)


def load_tokenizer(filename):
    f = open(filename, "rb")
    bpe = pickle.load(f)
    print(bpe.decode(bpe.encode(f"Testing tokenizer.")))
    return lambda x: bpe.encode(x), lambda xs: bpe.decode(xs), bpe.vocab_size
