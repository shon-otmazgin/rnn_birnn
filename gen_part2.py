import argparse
import random


def gen_01(n, file, palindrome):
    if n <= 0:
        return
    with open(file, 'w') as f:
        for i in range(n):
            s = []
            stop = random.random()          # generate random prob to stop
            while True:
                i = random.randint(0, 1)
                s.append('a' if i == 0 else 'b')
                if random.random() > stop:   # generate the same char
                    break
            if palindrome:
                f.write(''.join(s) + ''.join(s[::-1]) + '\n')
            else:
                f.write(''.join(s) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language generator')
    parser.add_argument("--n", type=int, default=0, help='Number of positive/negative examples (default: 0)')
    parser.add_argument("--suffix_file_name", type=str, default='_examples', help='pos_suffix / neg_suffix')
    args = parser.parse_args()

    gen_01(args.n, 'pos' + '_' + args.suffix_file_name, palindrome=True)
    gen_01(args.n, 'neg' + '_' + args.suffix_file_name, palindrome=False)
