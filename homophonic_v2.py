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

RECURSION_LIMIT = 1500  # default is 1000

ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]  # plain always lower case
EPSILON_NO_OCCURENCES = 1e-99  # zero has - infinite as log, must be << 1
EPSILON_DELTA_FLOAT = 0.000001  # to compare floats
EPSILON_PROBA = 1 / 1000  # to make sure we can give up searching


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
        self._log_freq_table = {q: math.log10(raw_frequency_table[q] / sum_occurences) for q in raw_frequency_table}

        self._worst_frequency = math.log10(EPSILON_NO_OCCURENCES / sum_occurences)

        after = time.time()
        elapsed = after - before
        print(f"N-Gram frequency file '{filename}' loaded in {elapsed:2.2f} seconds")

    @property
    def size(self) -> int:
        """ property """
        return self._size

    @property
    def log_freq_table(self) -> typing.Dict[str, float]:
        """ property """
        return self._log_freq_table

    @property
    def worst_frequency(self) -> float:
        """ property """
        return self._worst_frequency

    def __str__(self) -> str:
        """ for debug """
        return pprint.pformat(self._log_freq_table)


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

    def extracted_words(self, plain: str) -> typing.Tuple[float, typing.List[str]]:
        """ Tells the  of (more or less) plain text from the dictionary  """

        def words_probability(words: typing.List[str]) -> float:
            """ Quality of a list of word (probability) """
            return sum(map(lambda w: self._log_frequency_table.get(w, self._worst_frequency), words))

        @functools.lru_cache(maxsize=None)
        def splits(text: str) -> typing.List[typing.Tuple[str, str]]:
            """ All ways to split some text into a first word and remainder """
            return [(text[:cut + 1], text[cut + 1:]) for cut in range(min(self._longest_word, len(text)))]

        @functools.lru_cache(maxsize=None)
        def segment_rec(text: str) -> typing.List[str]:
            """ Best segmentation of text into words, by probability. """

            if not text:
                return list()

            candidates = [[first] + segment_rec(rest) for first, rest in splits(text)]
            return max(candidates, key=words_probability)

        best_words = segment_rec(plain)
        return words_probability(best_words), best_words

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

        # the cipher as it appears
        self._cipher_str = ''.join(self._content)

        # the different codes in cipher
        self._cipher_codes = ''.join(set(self._content))

        # list of cipher quadgrams with duplications
        cipher_quadgrams = [self._cipher_str[p: p + NGRAMS.size] for p in range(len(self._cipher_str) - (NGRAMS.size - 1))]

        # set of all quadgrams in cipher
        self._quadgrams_set = set(cipher_quadgrams)

        # a table where how many times quadgrams appear in cipher
        self._quadgrams_number_occurence_table = collections.Counter(cipher_quadgrams)

        # a table from a cipher to quadgrams that contain it
        self._quadgrams_localization_table: typing.Dict[str, typing.List[str]] = collections.defaultdict(list)
        for quadgram in self._quadgrams_set:
            for code in quadgram:
                self._quadgrams_localization_table[code].append(quadgram)

    def show_plain(self, selected_words: typing.List[str]) -> None:
        """ need to be here because this objects know jow many characters are missing """
        cipher_length = len(self._cipher_str)
        plain_length = sum(map(len, selected_words))
        coverage = plain_length / cipher_length * 100
        print(f"Covers {coverage:.2f}%")
        print(' '.join(selected_words))

    @property
    def cipher_codes(self) -> str:
        """ property """
        return self._cipher_codes

    @property
    def cipher_str(self) -> str:
        """ property """
        return self._cipher_str

    @property
    def quadgrams_set(self) -> typing.Set[str]:
        """ property """
        return self._quadgrams_set

    @property
    def quadgrams_number_occurence_table(self) -> typing.Counter[str]:
        """ property """
        return self._quadgrams_number_occurence_table

    @property
    def quadgrams_localization_table(self) -> typing.Dict[str, typing.List[str]]:
        """ property """
        return self._quadgrams_localization_table

    def __str__(self) -> str:
        return self._cipher_str


CIPHER: typing.Optional[Cipher] = None


class Decrypter:
    """ A decrypter : basically a dictionary cipher -> plain """

    def __init__(self) -> None:
        assert CIPHER is not None
        self._table = {c: '' for c in CIPHER.cipher_codes}

    def install(self, allocation: typing.Dict[str, str]) -> None:
        """ install an initial table """
        for cipher, plain in allocation.items():
            self._table[cipher] = plain

    def decode_some(self, cipher_part: str) -> str:
        """ decode """
        return ''.join(map(lambda c: self._table[c], cipher_part))

    def apply(self) -> str:
        """ apply the table (get a new plain from a cipher) """
        assert CIPHER is not None
        return self.decode_some(CIPHER.cipher_str)

    def swap(self, cipher1: str, cipher2: str) -> None:
        """ swap """
        self._table[cipher1], self._table[cipher2] = self._table[cipher2], self._table[cipher1]

    def __str__(self) -> str:
        return pprint.pformat(self._table)


DECRYPTER: typing.Optional[Decrypter] = None


class Attacker:
    """ Attacker """

    def __init__(self) -> None:

        assert DICTIONARY is not None
        assert CIPHER is not None
        assert DECRYPTER is not None
        assert NGRAMS is not None

        # make initial random allocation
        allocation = {k: random.choice(ALPHABET) for k in CIPHER.cipher_codes}

        # put it in crypter
        DECRYPTER.install(allocation)

        # a table for remembering frequencies
        self._quadgrams_frequency_quality_table: typing.Dict[str, float] = dict()

        # quadgram frequency quality table
        for quadgram in CIPHER.quadgrams_set:

            # plain
            plain = DECRYPTER.decode_some(quadgram)

            # remembered
            self._quadgrams_frequency_quality_table[quadgram] = NGRAMS.log_freq_table.get(plain, NGRAMS.worst_frequency) * CIPHER.quadgrams_number_occurence_table[quadgram]

        # quadgram overall frequency quality of cipher
        # summed
        self._overall_quadgrams_frequency_quality = sum(self._quadgrams_frequency_quality_table.values())

        if DEBUG:
            self._check_quadgram_frequency_quality()

    def _check_quadgram_frequency_quality(self) -> None:
        """ Evaluates quality from quadgram frequency DEBUG """

        assert DEBUG

        assert DECRYPTER is not None
        assert NGRAMS is not None

        # debug check
        qcheck = 0.
        plain = DECRYPTER.apply()
        for position in range(len(plain) - NGRAMS.size + 1):
            quadgram = plain[position: position + NGRAMS.size]
            qcheck += NGRAMS.log_freq_table[quadgram]

        assert abs(qcheck - self._overall_quadgrams_frequency_quality) < EPSILON_DELTA_FLOAT

    def _swap(self, cipher1: str, cipher2: str) -> None:
        """ swap """

        assert NGRAMS is not None
        assert CIPHER is not None
        assert DECRYPTER is not None

        DECRYPTER.swap(cipher1, cipher2)

        # effect

        for cipher in cipher1, cipher2:
            for quadgram in CIPHER.quadgrams_localization_table[cipher]:

                # value obliterated
                self._overall_quadgrams_frequency_quality -= self._quadgrams_frequency_quality_table[quadgram]

                # new plain
                plain = DECRYPTER.decode_some(quadgram)

                # new value
                new_value = NGRAMS.log_freq_table.get(plain, NGRAMS.worst_frequency) * CIPHER.quadgrams_number_occurence_table[quadgram]

                # remembered
                self._quadgrams_frequency_quality_table[quadgram] = new_value

                # summed
                self._overall_quadgrams_frequency_quality += new_value

    def _climb(self) -> bool:
        """ climb : try to improve things... """

        assert CIPHER is not None
        assert DECRYPTER is not None

        changes = [(cipher1, cipher2) for cipher1, cipher2 in itertools.combinations(CIPHER.cipher_codes, 2) if DECRYPTER.decode_some(cipher1) != DECRYPTER.decode_some(cipher2)]
        random.shuffle(changes)

        for cipher1, cipher2 in changes:

            # -----------------------
            #  does the change improve things ?
            # -----------------------

            # keep a note of quality before change
            old_overall_quadgrams_frequency_quality = self._overall_quadgrams_frequency_quality

            # apply change now
            self._swap(cipher1, cipher2)

            if DEBUG:
                self._check_quadgram_frequency_quality()

            # did the quality improve ?
            if self._overall_quadgrams_frequency_quality > old_overall_quadgrams_frequency_quality:
                # yes : stop looping : we have improved
                return True

            # no improvement so undo
            self._swap(cipher1, cipher2)

            if DEBUG:
                self._check_quadgram_frequency_quality()

            # restore value
            self._overall_quadgrams_frequency_quality = old_overall_quadgrams_frequency_quality

        return False

    def ascend(self) -> None:
        """ ascend : keep climbing until fails to do so """

        while True:

            # keeps climbing until fails to do so
            succeeded = self._climb()
            if not succeeded:
                print()
                return
            print("/", end='', flush=True)

    @property
    def overall_quadgrams_frequency_quality(self) -> float:
        """ property """
        return self._overall_quadgrams_frequency_quality


ATTACKER: typing.Optional[Attacker] = None


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
    global CIPHER
    CIPHER = Cipher(cipher_file, debug_mode)
    print(f"Cipher='{CIPHER}'")

    global DECRYPTER
    DECRYPTER = Decrypter()

    if debug_mode:
        identity = {c: c for c in CIPHER.cipher_codes}
        DECRYPTER.install(identity)
        plain = DECRYPTER.apply()
        dictionary_quality, selected_words = DICTIONARY.extracted_words(plain)
        print(f"{dictionary_quality=}")
        CIPHER.show_plain(selected_words)
        return

    best_trigram_quality_sofar: typing.Optional[float] = None
    while True:

        # start a new session
        global ATTACKER
        ATTACKER = Attacker()
        ATTACKER.ascend()

        # get dictionary quality of result
        clear = DECRYPTER.apply()
        dictionary_quality, selected_words = DICTIONARY.extracted_words(clear)

        if best_trigram_quality_sofar is None or ATTACKER.overall_quadgrams_frequency_quality > best_trigram_quality_sofar:
            print(f"{dictionary_quality=}")
            CIPHER.show_plain(selected_words)
            best_trigram_quality_sofar = ATTACKER.overall_quadgrams_frequency_quality


if __name__ == '__main__':

    sys.setrecursionlimit(RECURSION_LIMIT)

    # this if script too slow and profile it
    if PROFILE:
        PR = cProfile.Profile()
        PR.enable()

    # this to know how long it takes
    BEFORE = time.time()

    print("Press Ctrl+C to stop program")
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+C detected !")

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
