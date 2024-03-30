import model as m
import argparse

import tokenizer


def get_args():
    parser = argparse.ArgumentParser(
        prog='TransformerDemo',
        description='Simple transformer-based LLM')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-r', '--run', action='store_true')
    parser.add_argument('-c', '--context', type=int, required=True)
    parser.add_argument('-d', '--dimensions', type=int, required=True)
    parser.add_argument('-i', '--iterations', type=int)
    parser.add_argument('-f', '--filename', type=str, required=True)
    parser.add_argument('-o', '--output', type=int)
    parser.add_argument('-b', '--blocks', type=int)
    parser.add_argument('-k', '--tokenizer')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    m.set_params(args.dimensions, args.context, args.iterations, args.blocks)

    if args.tokenizer:
        enc, dec, voc = tokenizer.load_tokenizer(args.tokenizer)
        m.set_tokenizer(enc, dec, voc)
        print(f'Configured custom tokenizer {args.tokenizer} with vocab_size {voc}.')
    else:
        print('Using default character-based tokenizer.')

    m.init()
    model = m.Transformer().to(m.device)

    if args.run:
        model.load_state_dict(m.torch.load(args.filename))
        model.eval()
        while True:
            s = input("\n> ")
            for _ in range(args.output or 100):
                s = model.generate_char(s)
                print(s[-1], end="")
            print("\n")
    elif args.train:
        optimizer = m.torch.optim.AdamW(model.parameters(), lr=m.learning_rate)
        for iter in range(m.max_iters):
            if iter % m.eval_interval == 0:
                losses = m.estimate_loss(model)
                print(
                    f"""{int(iter / m.max_iters * 100.0)}% done. Training loss: {losses['train']:.4f}. Validation loss: {losses['val']:.4f}""")
            xb, yb = m.get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        m.torch.save(model.state_dict(), args.filename)
        print("Done training.")
