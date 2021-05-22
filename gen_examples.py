import argparse
import random


def gen_examples(reg, n, file):
    if n <= 0:
        return
    with open(file, 'w') as f:
        for i in range(n):
            s = []
            for g in reg:
                stop = random.random()                                          # generate random prob to stop
                while True:
                    s.append(g if g != r'\d' else str(random.randint(1, 9)))    # generate the same char
                    if random.random() <= stop:                                 # stop in range [0, stop]
                        break
            f.write(''.join(s) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language generator')
    parser.add_argument("N", type=int, default=500, help='Number of positive/negative examples (default: 0)')
    parser.add_argument("--suffix_file_name", type=str, default='examples', help='pos_suffix / neg_suffix defualt: _examples')
    args = parser.parse_args()

    n = args.N
    pos_file = 'pos' + '_' + args.suffix_file_name
    neg_file = 'neg' + '_' + args.suffix_file_name
    pos_reg = [r'\d', r'a', r'\d', r'b', r'\d', r'c', r'\d', r'd', r'\d']
    neg_reg = [r'\d', r'a', r'\d', r'c', r'\d', r'b', r'\d', r'd', r'\d']

    print(f'Generating {n} positive examples to {pos_file}')
    gen_examples(pos_reg, n, pos_file)
    print('Done.')

    print(f'Generating {n} negative examples to {neg_file}')
    gen_examples(neg_reg, n, neg_file)
    print('Done.')

