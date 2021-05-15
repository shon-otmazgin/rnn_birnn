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
    parser.add_argument("--pos", type=int, default=0, help='Number of positive examples (default: 0)')
    parser.add_argument("--pos_file", type=str, default='pos_examples', help='positive examples output file (default: pos_examples)')
    parser.add_argument("--neg", type=int, default=0, help='Number of negative examples (default: 0)')
    parser.add_argument("--neg_file", type=str, default='neg_examples', help='negative examples output file (default: neg_examples)')
    args = parser.parse_args()

    pos_reg = [r'\d', r'a', r'\d', r'b', r'\d', r'c', r'\d', r'd', r'\d']
    neg_reg = [r'\d', r'a', r'\d', r'c', r'\d', r'b', r'\d', r'd', r'\d']

    gen_examples(pos_reg, args.pos, args.pos_file)
    gen_examples(neg_reg, args.neg, args.neg_file)
