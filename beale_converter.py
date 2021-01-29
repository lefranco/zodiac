#!/usr/bin/env python3


"""
Input : text
Output : text
"""

import typing
import argparse
import copy

CRYPT_CHARACTERS = [chr(i) for i in range(ord('!'), ord('~') + 1)] + [chr(i) for i in range(ord('À'), ord('ÿ') + 1)]


class Cipher:
    """ A cipher : basically a string """

    def __init__(self, filename: str) -> None:

        self._convert_table: typing.Dict[int, str] = dict()
        available_ciphers = copy.copy(CRYPT_CHARACTERS)

        # the string (as a list) read from cipher file
        self._content: typing.List[str] = list()
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                for word in line.split():
                    try:
                        code = int(word)
                    except ValueError:
                        assert False, f"Failed to convert {word} to integer"
                    if code in self._convert_table:
                        cipher = self._convert_table[code]
                    else:
                        assert available_ciphers, "Out of ciphers for conversion"
                        cipher = available_ciphers.pop(0)
                        self._convert_table[code] = cipher
                    self._content.append(cipher)

        # the different codes in cipher
        self._cipher_codes = ''.join(sorted(set(self._content)))

    def print_difficulty(self) -> None:
        """ climb_difficulty """
        print(f"INFORMATION: We have a cipher with {len(self._cipher_codes)} different codes and a length of {len(self._content)}")

    @property
    def cipher_codes(self) -> str:
        """ property """
        return self._cipher_codes

    def __str__(self) -> str:
        return ''.join(self._content)


CIPHER: typing.Optional[Cipher] = None


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input file with beale like cipher (numbers)')
    parser.add_argument('-o', '--output', required=True, help='output a file with converted cipher')
    args = parser.parse_args()

    print(f"INFORMATION : We are limited to {len(CRYPT_CHARACTERS)} different ciphers")

    cipher_input_file = args.input
    cipher_output_file = args.output

    global CIPHER
    CIPHER = Cipher(cipher_input_file)

    CIPHER.print_difficulty()

    with open(cipher_output_file, 'w') as file_handle:
        print(CIPHER, file=file_handle)


if __name__ == '__main__':
    main()
