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
import typing
import math
import functools
import copy
import pprint
import secrets  # instead of random

import cProfile
import pstats

PROFILE = False
DEBUG = False

RECURSION_LIMIT = 1500  # default is 1000

ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]  # plain always lower case
EPSILON_NO_OCCURENCES = 1e-99  # zero has - infinite as log, must be << 1
EPSILON_DELTA_FLOAT = 0.000001  # to compare floats
EPSILON_PROBA = 1 / 100  # 99% = to make sure we can give up searching

MAX_STUFFING = 10


class Letters:
    """ Says the frequency of letters """

    def __init__(self, filename: str):

        raw_frequency_table: typing.Dict[str, int] = dict()
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                letter_read, letter_str = line.split()
                letter = letter_read.lower()
                frequency = int(letter_str)
                raw_frequency_table[letter] = frequency

        assert len(raw_frequency_table) == len(ALPHABET)

        sum_occurences = sum(raw_frequency_table.values())

        # for normal values
        self._freq_table = {q: raw_frequency_table[q] / sum_occurences for q in raw_frequency_table}

    @property
    def freq_table(self) -> typing.Dict[str, float]:
        """ property """
        return self._freq_table

    def __str__(self) -> str:
        """ for debug """
        return pprint.pformat(self._freq_table)


LETTERS: typing.Optional[Letters] = None


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

    def __init__(self, filename: str) -> None:

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

    def __init__(self, filename: str) -> None:

        assert NGRAMS is not None

        # the string (as a list) read from cipher file
        self._content: typing.List[str] = list()
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                for word in line.split():
                    for code in word:
                        self._content.append(code)

        # the cipher as it appears
        self._cipher_str = ''.join(self._content)

        # the different codes in cipher
        self._cipher_codes = ''.join(sorted(set(self._content)))

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
        self._table: typing.Dict[str, str] = dict()
        self._reverse_table: typing.Dict[str, typing.Set[str]] = collections.defaultdict(set)

    def instantiate(self, allocation: typing.Dict[str, str]) -> None:
        """ instantiate """
        for cipher, plain in allocation.items():
            self._table[cipher] = plain
            self._reverse_table[plain].add(cipher)

    def decode_some(self, cipher_part: str) -> str:
        """ decode """
        return ''.join([self._table[c] for c in cipher_part])

    def apply(self) -> str:
        """ apply the table (get a new plain from a cipher) """
        assert CIPHER is not None
        return self.decode_some(CIPHER.cipher_str)

    def swap(self, cipher1: str, cipher2: str) -> None:
        """ swap : this is the most consuming function here """

        # note the plains
        plain1 = self._table[cipher1]
        plain2 = self._table[cipher2]

        # just a little check
        # optimized away
        #  assert plain1 != plain2

        # swap
        self._table[cipher1], self._table[cipher2] = self._table[cipher2], self._table[cipher1]

        # move ciphers in table from plain

        # cipĥer1
        self._reverse_table[plain1].remove(cipher1)
        self._reverse_table[plain2].add(cipher1)

        # cipĥer2
        self._reverse_table[plain2].remove(cipher2)
        self._reverse_table[plain1].add(cipher2)

    def as_key(self) -> str:
        """ as_key """
        return ''.join([list(self._reverse_table[p])[0] if p in self._reverse_table else '-' for p in ALPHABET])

    @property
    def reverse_table(self) -> typing.Dict[str, typing.Set[str]]:
        """ property """
        return self._reverse_table

    def __str__(self) -> str:
        return "\n".join([pprint.pformat(self._table), pprint.pformat(self._reverse_table)])


DECRYPTER: typing.Optional[Decrypter] = None


class Bucket:
    """ A bucket : says how many ciphers could be allocated to every alphabet letter """

    def __init__(self, substitution_mode: bool) -> None:

        assert CIPHER is not None

        if substitution_mode:

            self._table = {ll: 1 for ll in ALPHABET}

        else:

            #  how many different codes
            number_codes = len(CIPHER.cipher_codes)
            self._table = {ll: 0 for ll in ALPHABET}

            while True:

                # criterion is deficit : how many I should have minus how many I have
                chosen = max(ALPHABET, key=lambda ll: LETTERS.freq_table[ll] * number_codes - self._table[ll])  # type: ignore
                self._table[chosen] += 1

                if sum(self._table.values()) == number_codes:
                    break

    def swap(self, letter1: str, letter2: str) -> None:
        """ swap """
        assert letter1 != letter2
        # TODO : swap letters in allocator

    @property
    def table(self) -> typing.Dict[str, int]:
        """ property """
        return self._table

    def __str__(self) -> str:
        return pprint.pformat(self._table)


BUCKET: typing.Optional[Bucket] = None


class Allocator:
    """ Alocator : Makes initial random allocation """

    def __init__(self, substitution_mode: bool) -> None:
        self._substitution_mode = substitution_mode

    def make_allocation(self) -> typing.Dict[str, str]:
        """ Makes a random allocation to start with """

        assert CIPHER is not None
        assert BUCKET is not None

        bucket = copy.copy(BUCKET.table)

        allocation: typing.Dict[str, str] = dict()

        for cipher in CIPHER.cipher_codes:
            letter_selected = secrets.choice([ll for ll in bucket if bucket[ll]])
            allocation[cipher] = letter_selected
            bucket[letter_selected] -= 1

        if self._substitution_mode and len(allocation) < len(ALPHABET):
            stuffing = sorted(set(ALPHABET) - set(allocation.values()))
            assert len(stuffing) <= MAX_STUFFING, "Too few different characters in substitution cipher"
            num = 0
            while True:
                letter_selected = secrets.choice(stuffing)
                stuffing.remove(letter_selected)
                allocation[f'{num}'] = letter_selected
                num += 1
                if not stuffing:
                    break

        return allocation


ALLOCATOR: typing.Optional[Allocator] = None
N_OPERATIONS = 0


class Attacker:
    """ Attacker """

    def __init__(self) -> None:

        assert NGRAMS is not None
        assert CIPHER is not None
        assert DECRYPTER is not None
        assert BUCKET is not None
        assert ALLOCATOR is not None

        initial_allocation = ALLOCATOR.make_allocation()
        DECRYPTER.instantiate(initial_allocation)

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
            qcheck += NGRAMS.log_freq_table.get(quadgram, NGRAMS.worst_frequency)

        assert abs(qcheck - self._overall_quadgrams_frequency_quality) < EPSILON_DELTA_FLOAT

    def _swap(self, cipher1: str, cipher2: str) -> None:
        """ swap: this is where most CPU time is spent in the program """

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

        # how many attempts before giving up ?
        number = len(ALPHABET) * (len(ALPHABET) - 1)
        attempts = int(math.log(EPSILON_PROBA) / math.log((number - 1) / number))

        while True:

            plain1 = secrets.choice(ALPHABET)
            plain2 = secrets.choice(sorted(set(ALPHABET) - set([plain1])))

            cipher1 = secrets.choice(list(DECRYPTER.reverse_table[plain1]))
            cipher2 = secrets.choice(list(DECRYPTER.reverse_table[plain2]))

            # -----------------------
            #  does the change improve things ?
            # -----------------------

            # keep a note of quality before change
            old_overall_quadgrams_frequency_quality = self._overall_quadgrams_frequency_quality

            # apply change now
            self._swap(cipher1, cipher2)

            # debug purpose only
            global N_OPERATIONS
            N_OPERATIONS += 1

            if DEBUG:
                self._check_quadgram_frequency_quality()

            # did the quality improve ?
            if self._overall_quadgrams_frequency_quality > old_overall_quadgrams_frequency_quality:
                # yes : stop looping : we have improved
                return True

            attempts -= 1
            if attempts == 0:
                return False

            # no improvement so undo
            self._swap(cipher1, cipher2)

            if DEBUG:
                self._check_quadgram_frequency_quality()

            # restore value
            self._overall_quadgrams_frequency_quality = old_overall_quadgrams_frequency_quality

    def ascend(self) -> None:
        """ ascend : keep climbing until fails to do so """

        while True:

            # keeps climbing until fails to do so
            succeeded = self._climb()
            if not succeeded:
                print("-", flush=True)
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
    parser.add_argument('-n', '--ngrams', required=True, help='input a file with frequency table for quadgrams (n-letters)')
    parser.add_argument('-d', '--dictionary', required=True, help='input a file with frequency table for words (dictionary) to use')
    parser.add_argument('-l', '--letters', required=True, help='input a file with frequency table for letters')
    parser.add_argument('-c', '--cipher', required=True, help='cipher to attack')
    parser.add_argument('-s', '--substitution_mode', required=False, help='cipher is simple substitution (not homophonic)', action='store_true')
    args = parser.parse_args()

    letters_file = args.letters
    global LETTERS
    LETTERS = Letters(letters_file)
    #  print(LETTERS)

    ngrams_file = args.ngrams
    global NGRAMS
    NGRAMS = Ngrams(ngrams_file)
    #  print(NGRAMS)

    dictionary_file = args.dictionary
    global DICTIONARY
    DICTIONARY = Dictionary(dictionary_file)
    #  print(DICTIONARY)

    cipher_file = args.cipher
    global CIPHER
    CIPHER = Cipher(cipher_file)
    #  print(f"Cipher='{CIPHER}'")

    global DECRYPTER
    DECRYPTER = Decrypter()

    substitution_mode = args.substitution_mode

    global BUCKET
    BUCKET = Bucket(substitution_mode)
    #  print(f"Bucket='{BUCKET}'")

    global ALLOCATOR
    ALLOCATOR = Allocator(substitution_mode)
    #  print(f"Allocator='{ALLOCATOR}'")

    global ATTACKER

    start_time = time.time()
    best_trigram_quality_sofar: typing.Optional[float] = None

    # outer hill climb
    while True:

        # inner hill climb
        while True:

            # start a new session
            ATTACKER = Attacker()
            ATTACKER.ascend()

            # get dictionary quality of result
            assert DECRYPTER is not None
            clear = DECRYPTER.apply()
            dictionary_quality, selected_words = DICTIONARY.extracted_words(clear)

            if best_trigram_quality_sofar is None or ATTACKER.overall_quadgrams_frequency_quality > best_trigram_quality_sofar:
                print(f"{dictionary_quality=}")
                CIPHER.show_plain(selected_words)
                now = time.time()
                speed = N_OPERATIONS / (now - start_time)
                print(f"{speed=}")
                key = DECRYPTER.as_key().upper()
                score = ATTACKER.overall_quadgrams_frequency_quality
                print(f"{key=} {score=}")
                best_trigram_quality_sofar = ATTACKER.overall_quadgrams_frequency_quality

            # TODO : stop at some point inner hill climb

        # TODO : change BUCKET for outer hill climb


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
