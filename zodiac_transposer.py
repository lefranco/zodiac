#!/usr/bin/env python3


"""
Input : text
Output : text
"""

import typing
import argparse
import collections

CRYPT_CHARACTERS = [chr(i) for i in range(ord('!'), ord('~') + 1)]

WIDTH_CIPHER = 17
HEIGHT_BLOCK = 9
SHIFT = 2


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
                        self._content.append(code)

        #  the different codes in cipher
        self._cipher_codes = ''.join(sorted(set(self._content)))

        self._blocks: typing.Dict[int, typing.Dict[typing.Tuple[int, int], str]] = collections.defaultdict(dict)
        self._transposed: typing.List[str] = list()

    def print_difficulty(self) -> None:
        """ climb_difficulty """
        print(f"INFORMATION: We have a cipher with {len(self._cipher_codes)} different codes and a length of {len(self._content)}")

    def transpose(self) -> None:
        """ Does the zodiac Z340 transposition to the cipher """

        # put in blocks
        for num, cipher in enumerate(self._content):
            column = num % WIDTH_CIPHER
            line = num // WIDTH_CIPHER
            line_in_block = line % HEIGHT_BLOCK
            num_block = line // HEIGHT_BLOCK
            self._blocks[num_block][(line_in_block, column)] = cipher

        # extract from blocks
        for num_block, content_block in self._blocks.items():

            # for a complete block
            if len(content_block) == HEIGHT_BLOCK * WIDTH_CIPHER:

                for pos_start in range(WIDTH_CIPHER):
                    position = pos_start

                    for cur_line in range(HEIGHT_BLOCK):
                        cur_col = (pos_start + (SHIFT * cur_line)) % WIDTH_CIPHER
                        cipher = self._blocks[num_block][(cur_line, cur_col)]
                        self._transposed.append(cipher)

                # just first block will be enough
                # the second hs two issues :
                # the upper left if fixed
                # one letter is ignored
                break

            # for the last incomplete block
            # note : it is completely mixed
            else:
                for position in range(HEIGHT_BLOCK * WIDTH_CIPHER):
                    cur_line = position // WIDTH_CIPHER
                    cur_col = position % WIDTH_CIPHER
                    if (cur_line, cur_col) not in self._blocks[num_block]:
                        break
                    cipher = self._blocks[num_block][(cur_line, cur_col)]
                    self._transposed.append(cipher)

    @property
    def cipher_codes(self) -> str:
        """ property """
        return self._cipher_codes

    def __str__(self) -> str:
        return ''.join(self._transposed)


CIPHER: typing.Optional[Cipher] = None


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input file with cipher')
    parser.add_argument('-o', '--output', required=True, help='output a file with cipher transposed')
    args = parser.parse_args()

    cipher_input_file = args.input
    cipher_output_file = args.output

    global CIPHER
    CIPHER = Cipher(cipher_input_file)

    CIPHER.print_difficulty()
    CIPHER.transpose()

    with open(cipher_output_file, 'w') as file_handle:
        print(CIPHER, file=file_handle)


if __name__ == '__main__':
    main()
