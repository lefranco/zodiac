#!/usr/bin/python3


"""
Input : text (file with words)
Output : code (file with characters)
"""

import typing
import argparse
import collections
import unicodedata
import contextlib


ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]


class Plain:
    """ A plain : basically a string """

    def __init__(self, filename: str) -> None:

        self._content: typing.List[str] = list()
        self._words: typing.List[str] = list()

        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                if line:
                    nfkd_form = unicodedata.normalize('NFKD', line)
                    only_ascii = nfkd_form.encode('ASCII', 'ignore')
                    only_ascii_str = only_ascii.decode()
                    for word in only_ascii_str.split():
                        word = word.lower()
                        word = ''.join([ll for ll in word if ll in ALPHABET])
                        self._words.append(word)
                        for letter in word:
                            assert letter in ALPHABET
                            self._content.append(letter)

        self._plain_str = ''.join(self._content)

    def stats(self, file_handle: typing.TextIO) -> None:
        """ stats """

        quads_count = collections.Counter(self._words)
#        quads_count = collections.Counter([self._plain_str[p: p+4] for p in range(len(self._plain_str)-4)])

        with contextlib.redirect_stdout(file_handle):
            for quad in sorted(quads_count, key=lambda q: quads_count[q], reverse=True):
                num = quads_count[quad]
                print(f"{quad} {num}")

    def __str__(self) -> str:
        return self._plain_str


PLAIN: typing.Optional[Plain]


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input file with plain (can have spaces within - will be removed - can have accents - will be corrected)')
    parser.add_argument('-o', '--output', required=False, help='quads')
    args = parser.parse_args()

    plain_input_file = args.input
    global PLAIN
    PLAIN = Plain(plain_input_file)

    plain_output_file = args.output
    if plain_output_file is not None:
        with open(plain_output_file, 'w') as file_handle:
            PLAIN.stats(file_handle)


if __name__ == '__main__':
    main()
