#!/usr/bin/env python3


"""
Input : text (file with words)
Output : code (file with characters)
"""

import typing
import argparse
import collections
import unicodedata
import pprint
import contextlib
import secrets  # instead of random

ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]
CRYPT_CHARACTERS = [chr(i) for i in range(ord('!'), ord('~') + 1)]


class Letters:
    """ Ngrams : says the frequency of letters """

    def __init__(self, filename: str):

        raw_frequency_table: typing.Dict[str, int] = dict()
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                letter_read, letter_str = line.split()
                letter = letter_read.lower()
                frequency = int(letter_str)
                raw_frequency_table[letter] = frequency

        assert len(raw_frequency_table) == len(ALPHABET)

        sum_occurences = sum(raw_frequency_table.values())

        # for normal values
        self._freq_table = {q: raw_frequency_table[q] / sum_occurences for q in raw_frequency_table}

    @property
    def freq_table(self) -> typing.Dict[str, float]:
        """ property """
        return self._freq_table

    def __str__(self) -> str:
        """ for debug """
        return pprint.pformat(self._freq_table)


LETTERS: typing.Optional[Letters] = None


class Crypter:
    """ A crypter : basically a dictionnary """

    def __init__(self, substitution_mode: bool, number: typing.Optional[int]):

        assert LETTERS is not None

        # how many cipher for a plain
        if substitution_mode:
            numbers = {ll: 1 for ll in ALPHABET}
            pool = set(map(lambda c: c.upper(), ALPHABET))
        else:
            assert number is not None
            numbers = {ll: max(1, round(LETTERS.freq_table[ll] * number) - 1 + secrets.choice(range(-1, 2))) for ll in ALPHABET}
            pool = set(CRYPT_CHARACTERS)
            assert len(pool) >= sum(numbers.values()), "Too many ciphers to create"

        # encrypting table
        self._table: typing.Dict[str, typing.List[str]] = collections.defaultdict(list)
        for letter in ALPHABET:
            for _ in range(numbers[letter]):
                code = secrets.choice(list(pool))
                pool.remove(code)
                self._table[letter].append(code)

    def encrypt(self, char: str) -> str:
        """ encode """
        return secrets.choice(self._table[char])

    def print_key(self, file_handle: typing.TextIO) -> None:
        """ print_key """

        assert CIPHER is not None

        with contextlib.redirect_stdout(file_handle):

            print("-" * len(ALPHABET))
            print(''.join(ALPHABET))
            most_affected = max([len(s) for s in self._table.values()])
            for rank in range(most_affected):
                for letter in ALPHABET:
                    if letter in CIPHER.clear_content:
                        ciphers = sorted(self._table[letter])
                        if rank < len(ciphers):
                            cipher = ciphers[rank]
                            print(cipher, end='')
                        else:
                            print(' ', end='')
                    else:
                        print(' ', end='')
                print()
            print("-" * len(ALPHABET))


CRYPTER: typing.Optional[Crypter]


class Cipher:
    """ A cipher : basically a string """

    def __init__(self, filename: str) -> None:

        assert CRYPTER is not None
        self._content: typing.List[str] = list()
        self._clear_content: typing.List[str] = list()

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
                            self._clear_content.append(letter)
                            code = CRYPTER.encrypt(letter)
                            self._content.append(code)

        self._cipher_str = ''.join(self._content)

    @property
    def clear_content(self) -> typing.List[str]:
        """ property """
        return self._clear_content

    def __str__(self) -> str:
        return self._cipher_str


CIPHER: typing.Optional[Cipher]


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--letters', required=True, help='input a file with frequency table for letters')
    parser.add_argument('-i', '--input', required=True, help='input file with plain (can have spaces within - will be removed - can have accents - will be corrected)')
    parser.add_argument('-s', '--substitution_mode', required=False, help='cipher is simple substitution (not homophonic)', action='store_true')
    parser.add_argument('-n', '--number', required=True, help='number of distinct characters to put in cipher if homophonic')
    parser.add_argument('-o', '--output', required=True, help='output a file with ciphers')
    parser.add_argument('-K', '--key_dump', required=False, help='dump crypter key to file')
    args = parser.parse_args()

    letters_file = args.letters
    global LETTERS
    LETTERS = Letters(letters_file)
    #  print(LETTERS)

    substitution_mode = args.substitution_mode
    number = int(args.number)

    if substitution_mode:
        assert number == len(ALPHABET), f"Number must be {len(ALPHABET)} for substitution_mode cipher"

    global CRYPTER
    CRYPTER = Crypter(substitution_mode, number)

    cipher_input_file = args.input
    cipher_output_file = args.output
    substitution_mode = args.substitution_mode
    global CIPHER
    CIPHER = Cipher(cipher_input_file)
    with open(cipher_output_file, 'w') as file_handle:
        print(CIPHER, file=file_handle)

    if args.key_dump:
        crypter_output_file = args.key_dump
        # will not print characters absent from cipher
        with open(crypter_output_file, 'w') as file_handle:
            CRYPTER.print_key(file_handle)


if __name__ == '__main__':
    main()
