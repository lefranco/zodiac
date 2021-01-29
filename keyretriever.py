#!/usr/bin/env python3


"""
Input : text (file with words) and code (file with characters)
Output : key
"""

import typing
import argparse
import collections
import unicodedata
import contextlib
import itertools
import sys

ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]
CRYPT_CHARACTERS = [chr(i) for i in range(ord('!'), ord('~') + 1)] + [chr(i) for i in range(ord('À'), ord('ÿ') + 1)]


class Plain:
    """ A cipher : basically a string """

    def __init__(self, filename: str) -> None:

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

        # the plain as it appears
        self._plain_str = ''.join(self._clear_content)

        # the different letters in plain
        self._plain_letters = ''.join(sorted(set(self._clear_content)))

    @property
    def clear_content(self) -> typing.List[str]:
        """ property """
        return self._clear_content

    @property
    def plain_letters(self) -> str:
        """ property """
        return self._plain_letters

    @property
    def plain_str(self) -> str:
        """ property """
        return self._plain_str

    def __str__(self) -> str:
        return self._plain_str


PLAIN: typing.Optional[Plain] = None


class Cipher:
    """ A cipher : basically a string """

    def __init__(self, filename: str) -> None:

        # the string (as a list) read from cipher file
        self._content: typing.List[str] = list()
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                for word in line.split():
                    for code in word:
                        assert code in CRYPT_CHARACTERS
                        self._content.append(code)

        # the cipher as it appears
        self._cipher_str = ''.join(self._content)

        # the different codes in cipher
        self._cipher_codes = ''.join(sorted(set(self._content)))

    @property
    def cipher_codes(self) -> str:
        """ property """
        return self._cipher_codes

    @property
    def cipher_str(self) -> str:
        """ property """
        return self._cipher_str

    def __str__(self) -> str:
        return self._cipher_str


CIPHER: typing.Optional[Cipher] = None


class Crypter:
    """ A crypter : basically a dictionnary """

    def __init__(self) -> None:

        assert PLAIN is not None
        assert CIPHER is not None

        # rebuild key
        self._table: typing.Dict[str, typing.Set[str]] = collections.defaultdict(set)
        self._reverse_table: typing.Dict[str, str] = dict()

        if len(PLAIN.plain_str) != len(CIPHER.cipher_str):
            print(PLAIN.plain_str)
            print(CIPHER.cipher_str)
            print("ERROR: Cipher and plain do not have same length")
            sys.exit(1)

        position = 0
        for plain, cipher in zip(PLAIN.plain_str, CIPHER.cipher_str):
            self._table[plain].add(cipher)

            if cipher not in self._reverse_table:
                self._reverse_table[cipher] = plain
            else:
                if self._reverse_table[cipher] != plain:

                    print("-" * position)
                    print(PLAIN.plain_str[:position + 1], end='')
                    print(" <<<")
                    print(CIPHER.cipher_str[:position + 1], end='')
                    print(" <<<")
                    print(f"ERROR: Cipher '{cipher}' is now '{plain}' but was previously '{self._reverse_table[cipher]}'")
                    sys.exit(1)

            position += 1

        # check
        for plain1, plain2 in itertools.combinations(PLAIN.plain_letters, 2):
            common = self._table[plain1] & self._table[plain2]
            common_show = ' '.join(common)
            assert not common, f"Conflict for plains {plain1} and {plain2} both encoded by ciphers {common_show}"

    def print_key(self, file_handle: typing.TextIO) -> None:
        """ print_key """

        assert PLAIN is not None

        with contextlib.redirect_stdout(file_handle):

            print("-" * len(ALPHABET))
            print(''.join(ALPHABET))
            most_affected = max([len(s) for s in self._table.values()])
            for rank in range(most_affected):
                for letter in ALPHABET:
                    if letter in PLAIN.plain_letters:
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


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plain_input', required=True, help='input file with plain (can have spaces within - will be removed - can have accents - will be corrected)')
    parser.add_argument('-c', '--cipher_input', required=True, help='input file with cipher')
    parser.add_argument('-K', '--key_dump', required=False, help='dump crypter key to file')
    args = parser.parse_args()

    # load plain
    plain_input_file = args.plain_input
    global PLAIN
    PLAIN = Plain(plain_input_file)
    #  print("Plain:")
    #  print(PLAIN)

    # load cipher
    cipĥer_input_file = args.cipher_input
    global CIPHER
    CIPHER = Cipher(cipĥer_input_file)
    #  print("Cipher:")
    #  print(CIPHER)

    # make correspondance
    global CRYPTER
    CRYPTER = Crypter()

    if args.key_dump:
        crypter_output_file = args.key_dump
        # will not print characters absent from cipher
        with open(crypter_output_file, 'w') as file_handle:
            CRYPTER.print_key(file_handle)


if __name__ == '__main__':
    main()
