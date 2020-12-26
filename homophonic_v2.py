#!/usr/bin/python3


"""
Decodes an 'homophonic cipher.
This is a cipher where :
  - one letter from plain text can be coded in one or more codes (usually to have even frequency
    of codes in cipher)
  - the seperation between words is not shown
"""

import sys
import time
import argparse
import collections
import itertools
import random
import typing
import math
import functools
import pprint

import cProfile
import pstats

PROFILE = False
DEBUG = False

# plain always lower case
ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]


EPSILON_NO_OCCURENCES = 1e-99  # zero has - infinite as log, must be << 1


class Ngrams:
    """ Ngrams : says the frequency of N grams (log (occurences / sum all) """

    def __init__(self, filename: str):

        before = time.time()
        self._size = 0

        raw_frequency_table: typing.Dict[str, int] = dict()
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                quadgram_read, frequency_str = line.split()
                quadgram = quadgram_read.lower()
                if self._size:
                    assert len(quadgram) == self._size
                else:
                    self._size = len(quadgram)
                frequency = int(frequency_str)
                raw_frequency_table[quadgram] = frequency

        coverage = (len(raw_frequency_table) / (len(ALPHABET) ** self._size)) * 100
        print(f"Frequency tables covers {coverage:.2f}% of possibilities")

        sum_occurences = sum(raw_frequency_table.values())

        # for normal values
        self._content = {q: math.log10(raw_frequency_table[q] / sum_occurences) for q in raw_frequency_table}

        # complete for absent values
        def_log_value = math.log10(EPSILON_NO_OCCURENCES / sum_occurences)
        self._content.update({''.join(letters): def_log_value for letters in itertools.product(ALPHABET, repeat=self._size) if ''.join(letters) not in self._content})

        after = time.time()
        elapsed = after - before
        print(f"N-Gram frequency file '{filename}' loaded in {elapsed:2.2f} seconds")

    @property
    def size(self) -> int:
        """ property """
        return self._size

    def __str__(self) -> str:
        """ for debug """
        return pprint.pformat(self._content)


NGRAMS: typing.Optional[Ngrams] = None


class Dictionary:
    """ Stores the list of word. Say how many words in tempted plain """

    def __init__(self, filename: str, limit: typing.Optional[int]) -> None:

        before = time.time()

        raw_frequency_table: typing.Dict[str, int] = dict()

        # pass one : read the file
        line_num = 1
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                word, frequency_str = line.split()
                word = word.lower()

                assert not [ll for ll in word if ll not in ALPHABET], f"ERROR : bad word found in dictionary line {line_num} : <{word}>"

                if word in raw_frequency_table:
                    print(f"WARNING : duplicated word '{word}' for dictionary line {line_num}")
                    continue

                frequency = int(frequency_str)
                raw_frequency_table[word] = frequency
                line_num += 1

                if limit is not None and len(raw_frequency_table) == limit:
                    break

        # pass two : enter data
        sum_occurences = sum(raw_frequency_table.values())
        self._log_frequency_table = {w: math.log10(raw_frequency_table[w] / sum_occurences) for w in raw_frequency_table}
        self._worst_frequency = math.log10(EPSILON_NO_OCCURENCES / sum_occurences)

        # longest word
        self._longest_word = max([len(w) for w in self._log_frequency_table])

        after = time.time()
        elapsed = after - before
        print(f"Word list file '{filename}' loaded in {elapsed:2.2f} seconds")


    def detected_words(self, plain: str) -> typing.List[str]:
        """ Tells the  of (more or less) plain text from the dictionary  """

        def words_probability(words: typing.Tuple[str]) -> float:
            """ Quality of a list of word (probability) """
            return sum(map(lambda w: self._log_frequency_table.get(w, self._worst_frequency), words))

        @functools.lru_cache(maxsize=None)
        def splits(text: str) -> typing.List[typing.Tuple[str, str]]:
            """ All ways to split some text into a first word and remainder """
            return [(text[:cut+1], text[cut+1:]) for cut in range(min(self._longest_word, len(text)))]

        @functools.lru_cache(maxsize=None)
        def segment_rec(text: str) -> typing.List[str]:
            """ Best segmentation of text into words, by probability. """

            if not text:
                return list()

            candidates = [[first] + segment_rec(rest) for first, rest in splits(text)]
            return max(candidates, key=words_probability)

        return segment_rec(plain)


    def __str__(self) -> str:
        """ for debug """
        return pprint.pformat(self._log_frequency_table)


DICTIONARY: typing.Optional[Dictionary] = None


class Cipher:
    """ A cipher : basically a string """

    def __init__(self, filename: str, debug_mode: bool) -> None:

        assert NGRAMS is not None

        # the string (as a list) read from cipher file
        self._content: typing.List[str] = list()
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                for word in line.split():
                    for code in word:
                        if debug_mode:
                            code = code.lower()
                            if code not in ALPHABET:
                                print(f"In debug mode ignored '{code}' from pseudo cipher - not in alphabet")
                                continue
                        self._content.append(code)

        # the different codes in cipher
        self._cipher_codes = list(set(self._content))

        # a table where how many times ngram appears
        self._cipher_str = ''.join(self._content)
        self._quadgram_number_occurence_table = collections.Counter([self._cipher_str[p: p + NGRAMS.size] for p in range(len(self._cipher_str) - (NGRAMS.size - 1))])

    @property
    def cipher_codes(self) -> typing.List[str]:
        """ property """
        return self._cipher_codes

    @property
    def cipher_str(self) -> str:
        """ property """
        return self._cipher_str

    @property
    def quadgram_number_occurence_table(self) -> typing.Counter[str]:
        """ property """
        return self._quadgram_number_occurence_table

    def __str__(self) -> str:
        return self._cipher_str


CIPHER: typing.Optional[Cipher] = None


class Decrypter:
    """ A decrypter : basically a dictionary cipher -> plain """

    def __init__(self, cipher: Cipher) -> None:
        self._cipher = cipher
        self._table = {c: '' for c in cipher.cipher_codes}

    def install(self, allocation: typing.Dict[str, str]) -> None:
        """ install an initial table """
        for cipher, plain in allocation.items():
            self._table[cipher] = plain

    def decode_one(self, cipher: str) -> str:
        """ decode """
        assert cipher in self._table
        return self._table[cipher]

    def apply(self) -> str:
        """ apply the table (get a new plain from a cipher) """
        return ''.join([self.decode_one(c) for c in self._cipher.cipher_str])

    def __str__(self) -> str:
        return pprint.pformat(self._table)


DECRYPTER: typing.Optional[Decrypter] = None


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', required=False, help='seed random generator to value')
    parser.add_argument('-n', '--ngrams', required=True, help='input a file with frequency table for quadgrams (n-letters)')
    parser.add_argument('-d', '--dictionary', required=True, help='input a file with frequency table for words (dictionary) to use')
    parser.add_argument('-l', '--limit', required=False, help='limit in number of words loaded from dictionnary')
    parser.add_argument('-c', '--cipher', required=True, help='cipher to attack')
    parser.add_argument('-D', '--debug', required=False, help='give altitude of cipher taken as plain obtained', action='store_true')
    args = parser.parse_args()

    #  seed = time.time()
    seed = args.seed
    if seed is None:
        seed = time.time()
    random.seed(seed)

    ngrams_file = args.ngrams
    global NGRAMS
    NGRAMS = Ngrams(ngrams_file)
    #  print(NGRAMS)

    dictionary_file = args.dictionary
    limit = int(args.limit) if args.limit is not None else None
    global DICTIONARY
    DICTIONARY = Dictionary(dictionary_file, limit)
    #  print(DICTIONARY)

    debug_mode = args.debug

    cipher_file = args.cipher
    cipher = Cipher(cipher_file, debug_mode)
    print(f"Cipher='{cipher}'")

    global DECRYPTER
    DECRYPTER = Decrypter(cipher)

    if debug_mode:
        identity = {c:c for c in cipher.cipher_codes}
        DECRYPTER.install(identity)
        plain = DECRYPTER.apply()
        detected_words = DICTIONARY.detected_words(plain)
        print(f"{detected_words=}")

    # attacker : TODO


if __name__ == '__main__':

    # this if script too slow and profile it
    if PROFILE:
        PR = cProfile.Profile()
        PR.enable()

    # this to know how long it takes
    BEFORE = time.time()
    main()
    AFTER = time.time()
    ELAPSED = AFTER - BEFORE
    #  how long it took
    print(f"{ELAPSED=:2.2f}")

    # stats
    if PROFILE:
        PR.disable()
        PS = pstats.Stats(PR)
        PS.strip_dirs()
        PS.sort_stats('time')
        PS.print_stats()  # uncomment to have profile stats

    sys.exit(0)
