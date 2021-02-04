#!/usr/bin/env python3


"""
Input : quads frequencies

After : sort the result with following copmmand
sort -n <outputfile> -k 2 -r > <sorted_file>

"""

import argparse
import typing
import math


def ngram_from_file(input_file: str, n_value: int) -> typing.Generator[typing.Tuple[str, float], None, None]:
    """ ngram_from_file """

    ngram = 'A' * n_value
    progress = 0
    with open(input_file, "rb") as file_handler:
        while True:

            # return stuff
            freq = file_handler.read(1)
            if freq == b'':
                return

            # get value
            value = ord(freq)
            if value != 0:
                yield ngram, int(math.exp(float(value) / 256.0) * 1000.)

            # show progress
            progress += 1
            if progress % 1000000 == 0:
                print(f"{(progress/(26**n_value)) * 100:0.2f}%", end=' ', flush=True)

            # increase
            ngram_l = list(ngram)
            cur_pos = n_value - 1
            while True:
                # increment
                ngram_l[cur_pos] = chr(ord(ngram_l[cur_pos]) + 1)
                # quit
                if ngram_l[cur_pos] != '[':
                    ngram = ''.join(ngram_l)
                    break
                # apply carry
                ngram_l[cur_pos] = 'A'
                cur_pos -= 1


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input file ngrams')
    parser.add_argument('-n', '--n_value', required=True, help='n value for ngrams')
    parser.add_argument('-o', '--output', required=True, help='output file ngrams')
    args = parser.parse_args()

    input_file = args.input
    n_value = int(args.n_value)

    output_file = args.output
    if output_file:
        with open(output_file, 'w') as file_handle:
            for ngram, freq in ngram_from_file(input_file, n_value):
                file_handle.write(f"{ngram} {freq}\n")


if __name__ == '__main__':
    main()
