#!/usr/bin/env python3


"""
Repairs a ngram file

"""

import time
import argparse
import collections
import typing
import unicodedata


class NgramsDict:
    """ NgramsDict """

    def __init__(self, filename: str, size: int):

        before = time.time()

        self._table: typing.Dict[str, int] = collections.defaultdict(int)
        with open(filename, encoding='utf-8') as filepointer:
            for num_line, line in enumerate(filepointer):
                line = line.rstrip('\n')
                nfkd_form = unicodedata.normalize('NFKD', line)
                only_ascii = nfkd_form.encode('ASCII', 'ignore')
                only_ascii_str = only_ascii.decode()
                entry_read, frequency_str = only_ascii_str.split()
                entry = entry_read.upper()
                if size and len(entry) != size:
                    print(f"Warning: N-Gram frequency file line {num_line} discarded (bad size)")
                    continue
                entry = entry.replace("'","")
                frequency = int(frequency_str)
                self._table[entry] += frequency  # n gram or dict entry may occur several times after removal of accents

        after = time.time()
        elapsed = after - before
        print(f"INFORMATION: N-Gram frequency file or Dictionnary '{filename}' loaded in {elapsed:2.2f} seconds")

    def __str__(self) -> str:
        """ for debug """
        return "\n".join([f"{k} {self._table[k]}" for k in sorted(self._table, key=lambda k: self._table[k], reverse=True)])


NGRAMS_DICT: typing.Optional[NgramsDict] = None


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_size', required=False, help='n of ngram if ngram file')
    parser.add_argument('-i', '--input_ngrams_dict', required=True, help='input a file with frequency table for n_grams (n-letters) or words')
    parser.add_argument('-o', '--output_ngrams_dict', required=True, help='same file repaired')
    args = parser.parse_args()

    if args.n_size is not None:
        ngrams_dict_size = int(args.n_size)
    else:
        ngrams_dict_size = 0
    ngrams_dict_file = args.input_ngrams_dict
    global NGRAMS_DICT
    NGRAMS_DICT = NgramsDict(ngrams_dict_file, ngrams_dict_size)
    #  print(NGRAMS_DICT)

    # file to output
    output_ngrams_dict_file = args.output_ngrams_dict
    with open(output_ngrams_dict_file, 'w') as file_handle:
        print(NGRAMS_DICT, file=file_handle)


if __name__ == '__main__':
    main()
