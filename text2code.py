#!/usr/bin/python3


"""
Input : text (file with words)
Output : code (file with characters)
"""

import typing
import argparse
import random
import collections
import unicodedata
import time

ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]
CRYPT_CHARACTERS = [chr(i) for i in range(ord('!'), ord('~') + 1)]


def load_frequency_table(filename: str) -> typing.Dict[str, int]:
    """ load_frequency_table """

    frequency_table: typing.Dict[str, int] = dict()
    with open(filename) as filepointer:
        for line in filepointer:
            line = line.rstrip('\n')
            letter_read, frequency_str = line.split()
            letter = letter_read.lower()
            frequency = int(frequency_str)
            frequency_table[letter] = frequency

    return frequency_table


class Crypter:
    """ A crypter : basically a dictionnary """

    def __init__(self, size: int, fake: bool):
        total = sum([LETTER_FREQUENCY_TABLE[ll] for ll in ALPHABET])
        numbers = {ll: max(1, (LETTER_FREQUENCY_TABLE[ll] * size) // total) for ll in ALPHABET}
        self._table: typing.Dict[str, typing.List[str]] = collections.defaultdict(list)

        if fake:
            for letter in ALPHABET:
                code = letter
                self._table[letter].append(code)
        else:
            pool = set(CRYPT_CHARACTERS)
            for letter in ALPHABET:
                for _ in range(numbers[letter]):
                    assert pool, "Not so many characters to encode with"
                    code = random.choice(list(pool))
                    pool.remove(code)
                    self._table[letter].append(code)

    def encode(self, char: str) -> str:
        """ encode """
        return random.choice(self._table[char])

    def __str__(self) -> str:
        return '\n'.join([f"{k} : {v}" for k, v in self._table.items()])


CRYPTER: typing.Optional[Crypter]


class Cipher:
    """ A cipher : basically a string """

    def __init__(self, filename: str) -> None:

        self._content: typing.List[str] = list()

        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                if line:
                    nfkd_form = unicodedata.normalize('NFKD', line)
                    only_ascii = nfkd_form.encode('ASCII', 'ignore')
                    only_ascii_str = only_ascii.decode()
                    for word in only_ascii_str.split():
                        word = word.lower()
                        for letter in word:
                            if letter not in ALPHABET:
                                continue
                            assert CRYPTER is not None
                            code = CRYPTER.encode(letter)
                            self._content.append(code)

        self._cipher_str = ''.join(self._content)

    def __str__(self) -> str:
        return self._cipher_str


LETTER_FREQUENCY_TABLE: typing.Dict[str, int] = dict()
CIPHER: typing.Optional[Cipher]


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frequency', required=True, help='input a file with frequency table for unigrams (letters)')
    parser.add_argument('-i', '--input', required=True, help='input file with cipher as words')
    parser.add_argument('-o', '--output', required=True, help='output a file with ciphers')
    parser.add_argument('-n', '--number', required=True, help='number of charactres in code')
    parser.add_argument('-d', '--dump', required=False, help='dump crypter to file')
    parser.add_argument('-F', '--fake', required=False, help='fake', action='store_true')
    args = parser.parse_args()

    seed = time.time()
    random.seed(seed)

    frequency_file = args.frequency
    global LETTER_FREQUENCY_TABLE
    LETTER_FREQUENCY_TABLE = load_frequency_table(frequency_file)

    global CRYPTER
    number = int(args.number)
    fake = args.fake
    CRYPTER = Crypter(number, fake)
    if args.dump:
        crypter_output_file = args.dump
        with open(crypter_output_file, 'w') as file_handle:
            print(CRYPTER, file=file_handle)

    cipher_input_file = args.input
    cipher_output_file = args.output
    global CIPHER
    CIPHER = Cipher(cipher_input_file)
    with open(cipher_output_file, 'w') as file_handle:
        print(CIPHER, file=file_handle)


if __name__ == '__main__':

    main()
