#!/usr/bin/env python3


"""
Repairs a ngram file

"""

import time
import argparse
import collections
import typing
import unicodedata


class Ngrams:
    """ Ngrams : says the frequency of N grams (log (occurences / sum all) """

    def __init__(self, filename: str, size: int):

        before = time.time()

        self._table: typing.Dict[str, int] = collections.defaultdict(int)
        with open(filename, encoding='utf-8') as filepointer:
            for num_line, line in enumerate(filepointer):
                line = line.rstrip('\n')
                nfkd_form = unicodedata.normalize('NFKD', line)
                only_ascii = nfkd_form.encode('ASCII', 'ignore')
                only_ascii_str = only_ascii.decode()
                n_gram_read, frequency_str = only_ascii_str.split()
                n_gram = n_gram_read
                if len(n_gram) != size:
                    continue
                frequency = int(frequency_str)
                self._table[n_gram] += frequency  # n gram may occur several times after removal of accents

        after = time.time()
        elapsed = after - before
        print(f"INFORMATION: N-Gram frequency file '{filename}' loaded in {elapsed:2.2f} seconds")

    def __str__(self) -> str:
        """ for debug """
        return "\n".join([f"{k} {self._table[k]}" for k in sorted(self._table, key=lambda k: self._table[k], reverse=True)])


NGRAMS: typing.Optional[Ngrams] = None


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_size', required=True, help='n of ngram')
    parser.add_argument('-i', '--input_ngrams', required=True, help='input a file with frequency table for n_grams (n-letters)')
    parser.add_argument('-o', '--output_ngrams', required=True, help='same file repaired')
    args = parser.parse_args()

    ngrams_size = int(args.n_size)
    ngrams_file = args.input_ngrams
    global NGRAMS
    NGRAMS = Ngrams(ngrams_file, ngrams_size)
    #  print(NGRAMS)

    # file to best solution online
    output_ngrams_file = args.output_ngrams
    with open(output_ngrams_file, 'w') as file_handle:
        print(NGRAMS, file=file_handle)


if __name__ == '__main__':
    main()
