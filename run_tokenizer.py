import argparse
import tokenizer


def get_args():
    parser = argparse.ArgumentParser(
        prog='BPETokenizer',
        description='Simple BPE tokenizer')
    parser.add_argument('-v', '--vocab', required=True)
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    tokenizer.train_tokenizer(int(args.vocab), args.input, args.output)
