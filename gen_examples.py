import argparse
import random


def gen_examples(reg, n, file):
    if n <= 0:
        return
    with open(file, 'w') as f:
        for i in range(n):
            s = []
            for g in reg:
                stop = random.random()          # generate random prob to stop
                while True:
                    s.append(g if g != r'\d' else str(random.randint(1, 9)))
                    if random.random() > stop:   # generate the same char
                        break
            f.write(''.join(s) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language generator')
    parser.add_argument("--n", type=int, default=0, help='Number of positive/negative examples (default: 0)')
    parser.add_argument("--suffix_file_name", type=str, default='_examples', help='pos_suffix / neg_suffix')
    args = parser.parse_args()

    pos_reg = [r'\d', r'a', r'\d', r'b', r'\d', r'c', r'\d', r'd', r'\d']
    neg_reg = [r'\d', r'a', r'\d', r'c', r'\d', r'b', r'\d', r'd', r'\d']

    gen_examples(pos_reg, args.n, 'pos' + '_' + args.suffix_file_name)
    gen_examples(neg_reg, args.n, 'neg' + '_' + args.suffix_file_name)
