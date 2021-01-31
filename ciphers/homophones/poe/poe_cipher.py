#!/usr/bin/env python3


"""
Input : text (file with words) and code (file with characters)
Output : key
"""

import typing
import argparse
import unicodedata
import collections
import contextlib
import sys
import itertools

import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]
CRYPT_CHARACTERS = [chr(i) for i in range(ord('!'), ord('~') + 1)] + [chr(i) for i in range(ord('À'), ord('ÿ') + 1)]

CHAR_WIDTH = 30
CHAR_HEIGHT = 50

PAGE_WIDTH = 1200
PAGE_HEIGHT = 2400

BIG_FONT_SIZE = 40
SMALL_FONT_SIZE = 30


class Plain:
    """ A cipher : basically a string """

    def __init__(self, filename: str) -> None:

        self._content: typing.List[typing.List[str]] = list()

        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                if not line:
                    continue
                line_plains: typing.List[str] = list()
                nfkd_form = unicodedata.normalize('NFKD', line)
                only_ascii = nfkd_form.encode('ASCII', 'ignore')
                only_ascii_str = only_ascii.decode()
                for word in only_ascii_str.split():
                    word = word.lower()
                    for letter in word:
                        if letter not in ALPHABET:
                            continue
                        line_plains.append(letter)
                self._content.append(line_plains)

    def display(self, img_page: typing.Any) -> None:
        """ output to old fashion text """

        def display_char(char: str) -> typing.Any:
            """ output to old fashion text """

            img_char = Image.new('RGB', (CHAR_WIDTH, CHAR_HEIGHT), color="white")
            draw = ImageDraw.Draw(img_char)
            font = ImageFont.truetype('/Library/Fonts/FreeSerif.ttf', (BIG_FONT_SIZE + SMALL_FONT_SIZE) // 2)
            width_font, height_font = draw.textsize(char, font=font)

            pos = ((CHAR_WIDTH - width_font) // 2, (CHAR_HEIGHT - height_font) // 2)

            draw.text(pos, char, fill="red", font=font)
            return img_char

        cur_pos = (0, CHAR_HEIGHT)
        for line_plains in self._content:
            for plain in line_plains:
                img_char = display_char(plain)
                assert cur_pos[0] + CHAR_WIDTH <= PAGE_WIDTH, "Display horizontal overflow"
                assert cur_pos[1] + CHAR_HEIGHT <= PAGE_HEIGHT, "Display vertical overflow"
                img_page.paste(img_char, cur_pos)
                cur_pos = (cur_pos[0] + CHAR_WIDTH, cur_pos[1])
            cur_pos = (0, cur_pos[1] + 2 * CHAR_HEIGHT)

    @property
    def content(self) -> typing.List[typing.List[str]]:
        """ property """
        return self._content

    def __str__(self) -> str:
        return '\n'.join([' '.join([str(c) for c in ll]) for ll in self._content])


PLAIN: typing.Optional[Plain] = None


class CipherRecord(typing.NamedTuple):
    """ A single cipher  """
    letter: str
    bigger: bool
    upside_down: bool

    def display(self) -> typing.Any:
        """ output to old fashion text """

        img_char = Image.new('RGB', (CHAR_WIDTH, CHAR_HEIGHT), color="white")
        draw = ImageDraw.Draw(img_char)
        font = ImageFont.truetype('/Library/Fonts/FreeSerif.ttf', BIG_FONT_SIZE if self.bigger else SMALL_FONT_SIZE)
        char = self.letter
        width_font, height_font = draw.textsize(char, font=font)

        pos = ((CHAR_WIDTH - width_font) // 2, (CHAR_HEIGHT - height_font) // 2)

        if self.upside_down:
            pos = pos[0], pos[1] - 10

        draw.text(pos, char, fill="black", font=font)
        if self.upside_down:
            img_char = img_char.rotate(180)
        return img_char

    def __str__(self) -> str:
        return f"{self.letter}{'+' if self.bigger else ''}{'u' if self.upside_down else ''}"


class Cipher:
    """ A cipher : basically a string """

    def __init__(self, filename: str) -> None:

        # cipher content : list of Cipher records
        self._content: typing.List[typing.List[CipherRecord]] = list()

        with open(filename) as filepointer:
            for num_line, line in enumerate(filepointer):
                line = line.rstrip('\n')
                if not line:
                    continue
                line_ciphers: typing.List[CipherRecord] = list()
                for word in line.split():
                    letter = word[0]
                    assert letter.lower() in ALPHABET, f"Problem line {num_line+1} : letter must be in alphabet"
                    assert len(word) <= 3, f"Problem line {num_line+1} : {word} limit to letter and possibly attributes"
                    if len(word) > 1:
                        attributes = word[1:]
                        assert all([a in ['+', 'u'] for a in attributes]), f"Problem line {num_line+1} : bad attributes: {attributes}"
                    else:
                        attributes = ''
                    cipher = CipherRecord(letter=letter, bigger='+' in attributes, upside_down='u' in attributes)
                    line_ciphers.append(cipher)
                self._content.append(line_ciphers)

    def display(self, img_page: typing.Any) -> None:
        """ output to old fashion text """

        cur_pos = (0, 0)
        for line_ciphers in self._content:
            for cipher in line_ciphers:
                img_char = cipher.display()
                assert cur_pos[0] + CHAR_WIDTH <= PAGE_WIDTH, "Display horizontal overflow"
                assert cur_pos[1] + CHAR_HEIGHT <= PAGE_HEIGHT, "Display vertical overflow"
                img_page.paste(img_char, cur_pos)
                cur_pos = (cur_pos[0] + CHAR_WIDTH, cur_pos[1])
            cur_pos = (0, cur_pos[1] + 2 * CHAR_HEIGHT)

    def output(self, file_handle: typing.TextIO) -> None:
        """ make cipher for solver """

        available = set(CRYPT_CHARACTERS)
        convert_table = dict()

        with contextlib.redirect_stdout(file_handle):
            for line_ciphers in self._content:
                for cipher in line_ciphers:
                    if cipher not in convert_table:
                        code = available.pop()
                        convert_table[cipher] = code

                    print(code, end=' ')
                print("\n")

    @property
    def content(self) -> typing.List[typing.List[CipherRecord]]:
        """ property """
        return self._content

    def __str__(self) -> str:
        return '\n'.join([' '.join([str(c) for c in ll]) for ll in self._content])


CIPHER: typing.Optional[Cipher] = None


class Crypter:
    """ A crypter : basically a dictionnary """

    def __init__(self) -> None:

        assert PLAIN is not None
        assert CIPHER is not None

        # rebuild key
        self._table: typing.Dict[str, typing.Set[CipherRecord]] = collections.defaultdict(set)
        self._reverse_table: typing.Dict[CipherRecord, str] = dict()

        plain_str = ''.join([''.join([str(c) for c in ll]) for ll in PLAIN.content])
        ciphers = [c for ll in CIPHER.content for c in ll]

        if len(plain_str) != len(ciphers):
            print(f"ERROR: Cipher len={len(plain_str)} and plain len={len(ciphers)} do not have same length")
            sys.exit(1)

        # need a mapping to inform user where the error is
        mapping = [(line, char_pos) for line, ll in enumerate(PLAIN.content) for char_pos, _ in enumerate(ll)]

        first_met: typing.Dict[CipherRecord, int] = dict()

        for pos, (plain, cipher) in enumerate(zip(plain_str, ciphers)):
            self._table[plain].add(cipher)

            if cipher not in self._reverse_table:
                self._reverse_table[cipher] = plain
                first_met[cipher] = pos
            else:
                should_be_plain = self._reverse_table[cipher]
                if should_be_plain != plain:

                    # print error
                    line, char_pos = mapping[pos]
                    first_pos = first_met[cipher]
                    first_line, first_char_pos = mapping[first_pos]
                    print(f"ERROR: Cipher '{cipher}' (line {line+1} character {char_pos+1}) is now '{plain}' but was previously '{self._reverse_table[cipher]}' (line {first_line+1} character {first_char_pos+1})")

                    # helps decision
                    matches = [p for p, c in zip(plain_str, ciphers) if c == cipher]
                    count = collections.Counter(matches)
                    print(f"Count : {count}")

                    sys.exit(1)

        # check
        plain_letters = {c for ll in PLAIN.content for c in ll}
        for plain1, plain2 in itertools.combinations(plain_letters, 2):
            common = self._table[plain1] & self._table[plain2]
            common_show = ' '.join([str(c) for c in common])
            assert not common, f"Conflict for plains {plain1} and {plain2} both encoded by ciphers {common_show}"


CRYPTER: typing.Optional[Crypter]


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plain_input', required=False, help='input file with plain (can have spaces within - will be removed - can have accents - will be corrected)')
    parser.add_argument('-c', '--cipher', required=True, help='input file with cipher in descriptive form')
    parser.add_argument('-d', '--dump', required=False, help='dump cipher and plain as picture to file')
    parser.add_argument('-o', '--output', required=False, help='output cipher to file in usuable form')
    args = parser.parse_args()

    # load plain
    plain_input_file = args.plain_input
    global PLAIN
    PLAIN = Plain(plain_input_file)
    #  print("Plain:")
    #  print(PLAIN)

    # load cipher
    cipher_input_file = args.cipher
    global CIPHER
    CIPHER = Cipher(cipher_input_file)
    #  print("Cipher:")
    #  print(CIPHER)

    cipher_dump_file = args.dump
    if cipher_dump_file:
        img_page = PIL.Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), color="white")
        CIPHER.display(img_page)
        PLAIN.display(img_page)
        img_page.save(cipher_dump_file)

    global CRYPTER
    CRYPTER = Crypter()

    cipher_output_file = args.output
    if cipher_output_file:
        with open(cipher_output_file, 'w') as file_handle:
            CIPHER.output(file_handle)


if __name__ == '__main__':
    main()
