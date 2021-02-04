#!/usr/bin/env python3


"""
Input : big text
Output : quads frequencies
"""

import typing
import argparse
import collections
import unicodedata
import contextlib


ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]


class Plain:
    """ A plain : list of words and list of letters """

    def __init__(self, filename: str) -> None:

        self._words: typing.List[str] = list()

        letters: typing.List[str] = list()
        with open(filename, encoding='utf-8') as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                if line:
                    # remove accents
                    nfkd_form = unicodedata.normalize('NFKD', line)
                    only_ascii = nfkd_form.encode('ASCII', 'ignore')
                    only_ascii_str = only_ascii.decode()
                    # remove all bujt alphabet and spaces
                    bad_chars = ''.join(list({ll for ll in only_ascii_str if not (ll in ALPHABET or ll == ' ')}))
                    my_table = only_ascii_str.maketrans(bad_chars, ' ' * len(bad_chars))
                    letters_spaces_only = only_ascii_str.translate(my_table)
                    # now we can work
                    for word in letters_spaces_only.split():
                        word = word.upper()
                        assert word
                        self._words.append(word)
                        for letter in word:
                            letters.append(letter)

        self._plain_str = ''.join(letters)

    def stats_ngrams(self, file_handle: typing.TextIO, n_value: int) -> None:
        """ stats """

        # counting  ngrams
        ngrams_count = collections.Counter([self._plain_str[p: p + n_value] for p in range(len(self._plain_str) - n_value + 1)])

        with contextlib.redirect_stdout(file_handle):
            for ngram in sorted(ngrams_count, key=lambda n: ngrams_count[n], reverse=True):
                num = ngrams_count[ngram]
                print(f"{ngram} {num}")

    def stats_dict(self, file_handle: typing.TextIO) -> None:
        """ stats """

        # counting  ngrams
        word_count = collections.Counter(self._words)

        with contextlib.redirect_stdout(file_handle):
            for word in sorted(word_count, key=lambda w: word_count[w], reverse=True):
                num = word_count[word]
                print(f"{word} {num}")


PLAIN: typing.Optional[Plain]


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input file with plain (can have spaces within - will be removed - can have accents - will be corrected)')
    parser.add_argument('-n', '--n_value', required=False, help='n value for ngrams')
    parser.add_argument('-g', '--ngrams_output', required=False, help='output ngrams frequency file')
    parser.add_argument('-d', '--dict_output', required=False, help='output word frequency file')
    args = parser.parse_args()

    plain_input_file = args.input
    global PLAIN
    PLAIN = Plain(plain_input_file)

    ngrams_output_output_file = args.ngrams_output
    if ngrams_output_output_file is not None:
        assert args.n_value is not None, "Need a n value for ngrams"
        n_value = int(args.n_value)
        assert 1 <= n_value <= 8, "Incorrect value for ngrams"
        with open(ngrams_output_output_file, 'w') as file_handle:
            PLAIN.stats_ngrams(file_handle, n_value)

    dict_output_output_file = args.dict_output
    if dict_output_output_file is not None:
        with open(dict_output_output_file, 'w') as file_handle:
            PLAIN.stats_dict(file_handle)


if __name__ == '__main__':
    main()
