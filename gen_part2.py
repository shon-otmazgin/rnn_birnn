import argparse
import random


def isPalindrome(s):
    ''' check if a number is a Palindrome '''
    s = str(s)
    return s == s[::-1]


def gen_01():
    s = []
    while True:
        i = random.randint(1, 9)
        s.append(str(i))
        if random.random() < 0.01:  # generate the same char
            break
    return ''.join(s)


def gen_palindrome_examples(n, file, palindrome):
    if n <= 0:
        return
    with open(file, 'w') as f:
        for i in range(n):
            s = gen_01()
            while isPalindrome(s):
                s = gen_01()
            if palindrome:
                mid = (len(s) // 2) + 1
                s = s[:mid]
                f.write(s + s[::-1] + '\n')
            else:
                f.write(s + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language generator')
    parser.add_argument("--n", type=int, default=0, help='Number of positive/negative examples (default: 0)')
    parser.add_argument("--suffix_file_name", type=str, default='_examples', help='pos_suffix / neg_suffix')
    args = parser.parse_args()

    gen_palindrome_examples(args.n, 'pos' + '_' + args.suffix_file_name, palindrome=True)
    gen_palindrome_examples(args.n, 'neg' + '_' + args.suffix_file_name, palindrome=False)
