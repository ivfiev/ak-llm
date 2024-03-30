from tiktoken._educational import SimpleBytePairEncoding
import pickle

gpt2_pattern = (
    r"""'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def train_tokenizer(vocab_size, input, output):
    i = open(input, "r")
    o = open(output, "wb")
    bpe = SimpleBytePairEncoding.train(i.read(), vocab_size, gpt2_pattern)
    bpe.vocab_size = vocab_size
    pickle.dump(bpe, o)


def load_tokenizer(filename):
    f = open(filename, "rb")
    bpe = pickle.load(f)
    print(bpe.decode(bpe.encode(f"Testing tokenizer {filename}.", None)))
    return lambda x: bpe.encode(x, None), lambda xs: bpe.decode(xs), bpe.vocab_size
