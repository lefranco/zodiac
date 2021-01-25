#!/usr/bin/env python3

# issue with multipocessing ?
#   !/usr/bin/pypy3


"""
Decodes an 'substition cipher.
This is a cipher where :
  - one letter from plain text can be coded in one codes (usually to have even frequency
    of codes in cipher)
  - the separation between words is not shown
"""

import sys
import time
import argparse
import collections
import typing
import math
import functools
import itertools
import copy
import contextlib
import pprint
import random

import cProfile
import pstats

PROFILE = False
DEBUG = False
IMPATIENT = False
VERBOSE = False


RECURSION_LIMIT = 5000  # default is 1000 - this is only for when putting spaces when displaying plain text
VERY_BAD_DICTIONNARY = - 1e99

ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]  # plain always lower case
EPSILON_NO_OCCURENCES = 1e-99  # zero has - infinite as log, must be << 1
EPSILON_DELTA_FLOAT = 0.000001  # to compare floats (for debug check)

# May be suppressed later on
MAX_ATTACKER_CLIMBS = 1

MAX_SUBSTITUTION_STUFFING = 10

K_TEMPERATURE_ZERO = 1000.   # by convention keep it that way
K_TEMPERATURE_REDUCTION = 0.05   # tuned ! - less : too slow - more : not efficient
K_TEMPERATURE_FACTOR = 0.5   # tuned ! - more : more tolerant when going down

REF_IOC = 0.


def load_reference_coincidence_index(filename: str) -> None:
    """ Loads IOC from file """
    with open(filename) as filepointer:
        for line in filepointer:
            line = line.rstrip('\n')
            global REF_IOC
            REF_IOC = float(line)
            print(f"INFORMATION: Reference IOC is {REF_IOC}")
            return


class Letters:
    """ Says the frequency of letters """

    def __init__(self, filename: str):

        raw_frequency_table: typing.Dict[str, int] = dict()
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                letter_read, letter_str = line.split()
                letter = letter_read.lower()
                assert letter in ALPHABET, "Problem with letters frequencies file content"
                frequency = int(letter_str)
                raw_frequency_table[letter] = frequency

        assert len(raw_frequency_table) == len(ALPHABET), "Problem with letters frequencies file content"

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

        raw_frequency_table: typing.Dict[str, int] = collections.defaultdict(int)
        with open(filename) as filepointer:
            for num_line, line in enumerate(filepointer):
                line = line.rstrip('\n')
                n_gram_read, frequency_str = line.split()
                n_gram = n_gram_read.lower()
                if self._size:
                    assert len(n_gram) == self._size, f"Problem with ngram file content line {num_line+1}"
                else:
                    self._size = len(n_gram)
                    print(f"INFORMATION: Using N-Grams with N={self._size}")
                frequency = int(frequency_str)
                raw_frequency_table[n_gram] += frequency  # n gram may occur several times after removal of accents

        coverage = (len(raw_frequency_table) / (len(ALPHABET) ** self._size)) * 100
        print(f"INFORMATION: Frequency tables covers {coverage:.2f}% of possibilities")

        sum_occurences = sum(raw_frequency_table.values())

        # for normal values
        self._log_freq_table = {q: math.log10(raw_frequency_table[q] / sum_occurences) for q in raw_frequency_table}

        self._worst_frequency = math.log10(EPSILON_NO_OCCURENCES / sum_occurences)

        after = time.time()
        elapsed = after - before
        print(f"INFORMATION: N-Gram frequency file '{filename}' loaded in {elapsed:2.2f} seconds")

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
        with open(filename) as filepointer:
            for line_num, line in enumerate(filepointer):
                line = line.rstrip('\n')
                word, frequency_str = line.split()
                word = word.lower()

                assert not [ll for ll in word if ll not in ALPHABET], f"ERROR : bad word found in dictionary line {line_num+1} : <{word}>"

                if word in raw_frequency_table:
                    print(f"WARNING : duplicated word '{word}' for dictionary line {line_num+1}")
                    continue

                frequency = int(frequency_str)
                raw_frequency_table[word] = frequency

                if limit is not None and len(raw_frequency_table) >= limit:
                    print(f"INFORMATION: Ignoring dictionary words after the {limit}th")
                    break

        # pass two : enter data
        sum_occurences = sum(raw_frequency_table.values())
        self._log_frequency_table = {w: math.log10(raw_frequency_table[w] / sum_occurences) for w in raw_frequency_table}
        self._worst_frequency = math.log10(EPSILON_NO_OCCURENCES / sum_occurences)

        # longest word
        self._longest_word_size = max([len(w) for w in self._log_frequency_table])

        after = time.time()
        elapsed = after - before
        print(f"INFORMATION: Dictionary (word list) file '{filename}' loaded in {elapsed:2.2f} seconds")

    def extracted_words(self, plain: str) -> typing.Tuple[float, typing.List[str]]:
        """ Tells the  of (more or less) plain text from the dictionary  """

        def words_probability(words: typing.List[str]) -> float:
            """ Quality of a list of word (probability) """
            return sum(map(lambda w: self._log_frequency_table.get(w, self._worst_frequency), words))

        @functools.lru_cache(maxsize=None)
        def splits(text: str) -> typing.List[typing.Tuple[str, str]]:
            """ All ways to split some text into a first word and remainder """
            return [(text[:cut + 1], text[cut + 1:]) for cut in range(min(self._longest_word_size, len(text)))]

        @functools.lru_cache(maxsize=None)
        def segment_rec(text: str) -> typing.List[str]:
            """ Best segmentation of text into words, by probability. """

            if not text:
                return list()

            candidates = [[first] + segment_rec(rest) for first, rest in splits(text)]
            return max(candidates, key=words_probability)

        try:
            best_words = segment_rec(plain)
        except RecursionError:
            return VERY_BAD_DICTIONNARY, ["(garbage presumed)"]
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
                        # substituion mode : check in alphabet, store as upper case
                        assert code.lower() in ALPHABET, f"Problem in substituion cipher with code {code}"
                        code = code.upper()
                        self._content.append(code)

        # the cipher as it appears
        self._cipher_str = ''.join(self._content)

        # the different codes in cipher
        self._cipher_codes = ''.join(sorted(set(self._content)))

        # list of cipher n_grams with duplications
        cipher_n_grams = [self._cipher_str[p: p + NGRAMS.size] for p in range(len(self._cipher_str) - (NGRAMS.size - 1))]

        # set of all n_grams in cipher
        self._n_grams_set = set(cipher_n_grams)

        # a table where how many times n_grams appear in cipher
        self._n_grams_number_occurence_table = collections.Counter(cipher_n_grams)

        # a table from a cipher to n_grams that contain it
        self._n_grams_localization_table: typing.Dict[str, typing.List[str]] = collections.defaultdict(list)
        for n_gram in self._n_grams_set:
            for code in n_gram:
                self._n_grams_localization_table[code].append(n_gram)

        # a table where how many times code appear in cipher
        self._codes_number_occurence_table = collections.Counter(self._content)

    def print_difficulty(self) -> None:
        """ climb_difficulty """
        print(f"INFORMATION: We have a cipher with {len(self._cipher_codes)} different codes and a length of {len(self._content)}")

    @property
    def cipher_codes(self) -> str:
        """ property """
        return self._cipher_codes

    @property
    def cipher_str(self) -> str:
        """ property """
        return self._cipher_str

    @property
    def n_grams_set(self) -> typing.Set[str]:
        """ property """
        return self._n_grams_set

    @property
    def n_grams_number_occurence_table(self) -> typing.Counter[str]:
        """ property """
        return self._n_grams_number_occurence_table

    @property
    def n_grams_localization_table(self) -> typing.Dict[str, typing.List[str]]:
        """ property """
        return self._n_grams_localization_table

    def __str__(self) -> str:
        return self._cipher_str


CIPHER: typing.Optional[Cipher] = None


class Decrypter:
    """ A decrypter : basically a dictionary cipher -> plain """

    def __init__(self) -> None:
        self._table: typing.Dict[str, str] = dict()
        self._reverse_table: typing.Dict[str, typing.Set[str]] = collections.defaultdict(set)
        self._allocated: typing.Set[str] = set()

    def instantiate(self, allocation: typing.Dict[str, str]) -> None:
        """ instantiate """
        self._reverse_table.clear()
        for cipher, plain in allocation.items():
            self._table[cipher] = plain
            self._reverse_table[plain].add(cipher)
        self._allocated = set(allocation.values())

    def allocation(self) -> typing.Dict[str, str]:
        """ exports allocation to rebuild crypter later on """
        return copy.deepcopy(self._table)

    def decode_some(self, cipher_part: str) -> str:
        """ decode """
        return ''.join([self._table[c] for c in cipher_part])

    def apply(self) -> str:
        """ apply the table (get a new plain from a cipher) """
        assert CIPHER is not None
        return self.decode_some(CIPHER.cipher_str)

    def swap(self, cipher1: str, cipher2: str) -> None:
        """ swap  """

        # note the plains
        plain1 = self._table[cipher1]
        plain2 = self._table[cipher2]

        # just a little check
        assert plain1 != plain2, "Internal error"

        # swap
        self._table[cipher1], self._table[cipher2] = self._table[cipher2], self._table[cipher1]

        # move ciphers in table from plain

        # cipĥer1
        self._reverse_table[plain1].remove(cipher1)
        self._reverse_table[plain2].add(cipher1)

        # cipĥer2
        self._reverse_table[plain2].remove(cipher2)
        self._reverse_table[plain1].add(cipher2)

    def print_key(self, file_handle: typing.TextIO) -> None:
        """ print_key """

        with contextlib.redirect_stdout(file_handle):
            print("-" * len(ALPHABET))
            print(''.join(ALPHABET))
            most_affected = max([len(s) for s in self._reverse_table.values()])
            for rank in range(most_affected):
                for letter in ALPHABET:
                    ciphers = sorted(self._reverse_table[letter])
                    if rank < len(ciphers):
                        cipher = ciphers[rank]
                        print(cipher, end='')
                    else:
                        print(' ', end='')
                print()
            print("-" * len(ALPHABET))

    @property
    def allocated(self) -> typing.Set[str]:
        """ property """
        return self._allocated

    @property
    def reverse_table(self) -> typing.Dict[str, typing.Set[str]]:
        """ property """
        return self._reverse_table

    def __str__(self) -> str:
        return "\n".join([pprint.pformat(self._table), pprint.pformat(self._reverse_table)])


DECRYPTER: typing.Optional[Decrypter] = None


def make_random_key() -> typing.Dict[str, str]:
    """ Makes a random key to start with """

    assert CIPHER is not None

    allocation: typing.Dict[str, str] = dict()
    pot = copy.copy(ALPHABET)

    for cipher in CIPHER.cipher_codes:
        letter_selected = random.choice(pot)
        pot.remove(letter_selected)
        allocation[cipher] = letter_selected

    if len(allocation) < len(ALPHABET):
        stuffing = sorted(set(ALPHABET) - set(allocation.values()))
        assert len(stuffing) <= MAX_SUBSTITUTION_STUFFING, "Too few different characters in substitution cipher"
        num = 0
        while True:
            letter_selected = random.choice(stuffing)
            stuffing.remove(letter_selected)
            allocation[f'{num}'] = letter_selected
            num += 1
            if not stuffing:
                break

    return allocation


class Evaluation:
    """ Evaluation """

    def __init__(self, n_grams_frequency_quality: float) -> None:
        self._n_grams_frequency_quality = n_grams_frequency_quality

    @property
    def n_grams_frequency_quality(self) -> float:
        """ property """
        return self._n_grams_frequency_quality

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Evaluation):
            return NotImplemented
        return abs(self._n_grams_frequency_quality - other.n_grams_frequency_quality) < EPSILON_DELTA_FLOAT

    def __gt__(self, other: 'Evaluation') -> bool:
        return self._n_grams_frequency_quality > other.n_grams_frequency_quality + EPSILON_DELTA_FLOAT

    def __str__(self) -> str:
        return f"ngram qual={self._n_grams_frequency_quality}"


class Solution:
    """ A solution """

    def __init__(self, quality: Evaluation, the_key: typing.Dict[str, str]) -> None:

        self._quality = quality
        self._the_key = copy.deepcopy(the_key)
        now = time.time()
        self._time_taken = now - START

    def print_solution(self, file_handle: typing.TextIO) -> None:
        """ print_solution """

        assert DICTIONARY is not None
        assert CIPHER is not None

        # get plain
        my_decrypter = Decrypter()
        my_decrypter.instantiate(self._the_key)
        plain = my_decrypter.apply()

        # get dictionary quality of result
        dictionary_quality, selected_words = DICTIONARY.extracted_words(plain)

        # get OIC of result
        nb_occ_plain = {ll: plain.count(ll) for ll in ALPHABET}
        overall_coincidence_index_quality = sum([nb_occ_plain[p] * (nb_occ_plain[p] - 1) for p in nb_occ_plain])
        index_of_coincidence = (len(ALPHABET) * overall_coincidence_index_quality) / (len(CIPHER.cipher_str) * (len(CIPHER.cipher_str) - 1))

        # print stuff
        print("=" * 50, file=file_handle)
        print(' '.join(selected_words), file=file_handle)
        print("=" * 50, file=file_handle)
        print(f"index_of_coincidence={index_of_coincidence} (reference={REF_IOC})", file=file_handle)
        print(f"dictionary_quality={dictionary_quality}", file=file_handle)
        print(f"quality={self._quality}", file=file_handle)
        print(f"time taken={self._time_taken} sec.", file=file_handle)
        my_decrypter.print_key(file_handle)


class Attacker:
    """ Attacker """

    def __init__(self) -> None:

        assert CIPHER is not None

        # a table for remembering frequencies
        self._n_grams_frequency_quality_table: typing.Dict[str, float] = dict()

        self._overall_n_grams_frequency_quality = 0.

        CIPHER.print_difficulty()

        # to measure speed
        self._n_operations = 0
        self._time_climbs_starts: typing.Optional[float] = None

        # simulated annealing
        self._temperature = 0.

    def _check_n_gram_frequency_quality(self) -> None:
        """ Evaluates quality from n_gram frequency DEBUG """

        assert DEBUG

        assert DECRYPTER is not None
        assert NGRAMS is not None

        # debug check
        qcheck = 0.
        plain = DECRYPTER.apply()
        for position in range(len(plain) - NGRAMS.size + 1):
            n_gram = plain[position: position + NGRAMS.size]
            qcheck += NGRAMS.log_freq_table.get(n_gram, NGRAMS.worst_frequency)

        assert abs(qcheck - self._overall_n_grams_frequency_quality) < EPSILON_DELTA_FLOAT, "Debug mode detected an error for N-Gram freq"

    def _swap(self, cipher1: str, cipher2: str) -> None:
        """ swap: this is where most CPU time is spent in the program """

        assert NGRAMS is not None
        assert CIPHER is not None
        assert DECRYPTER is not None

        DECRYPTER.swap(cipher1, cipher2)

        # effect

        for cipher in cipher1, cipher2:
            for n_gram in CIPHER.n_grams_localization_table[cipher]:

                # value obliterated
                self._overall_n_grams_frequency_quality -= self._n_grams_frequency_quality_table[n_gram]

                # new plain
                plain = DECRYPTER.decode_some(n_gram)

                # new value
                new_value = NGRAMS.log_freq_table.get(plain, NGRAMS.worst_frequency) * CIPHER.n_grams_number_occurence_table[n_gram]

                # remembered
                self._n_grams_frequency_quality_table[n_gram] = new_value

                # summed
                self._overall_n_grams_frequency_quality += new_value

        # to measure speed
        self._n_operations += 1

    def _go_slightly_down(self) -> bool:
        """ go slightly down : try not to give up by going slightly down after going up has failed... """

        def decide_accept(proba_acceptance: float) -> bool:
            """ decide_accept """
            assert 0. <= proba_acceptance <= 1.
            return random.random() <= proba_acceptance

        assert DECRYPTER is not None

        neighbours = {(c1, c2) for (p1, p2) in itertools.combinations(DECRYPTER.allocated, 2) for c1 in DECRYPTER.reverse_table[p1] for c2 in DECRYPTER.reverse_table[p2]}

        while True:

            # take a random neighbour
            neighbour = random.choice(list(neighbours))
            cipher1, cipher2 = neighbour

            # keep a note of quality before change
            old_overall_n_grams_frequency_quality = self._overall_n_grams_frequency_quality

            # apply change now
            self._swap(cipher1, cipher2)

            # quality should lower in this context
            assert self._overall_n_grams_frequency_quality <= old_overall_n_grams_frequency_quality
            delta_quality_percent = abs((self._overall_n_grams_frequency_quality - old_overall_n_grams_frequency_quality) / old_overall_n_grams_frequency_quality)
            proba_acceptance = math.exp(- delta_quality_percent / (K_TEMPERATURE_FACTOR * self._temperature))

            # apply acceptance probability function
            if decide_accept(proba_acceptance):
                self._temperature -= (K_TEMPERATURE_REDUCTION * self._temperature)
                return True

            # not selected so undo
            self._swap(cipher1, cipher2)
            return False

    def _go_up(self) -> bool:
        """ go up : try to improve things... """

        assert CIPHER is not None
        assert DECRYPTER is not None

        neighbours = {(c1, c2) for (p1, p2) in itertools.combinations(DECRYPTER.allocated, 2) for c1 in DECRYPTER.reverse_table[p1] for c2 in DECRYPTER.reverse_table[p2]}

        while True:

            # take a random neighbour
            neighbour = random.choice(list(neighbours))
            neighbours.remove(neighbour)
            cipher1, cipher2 = neighbour

            # -----------------------
            #  does the change improve things ?
            # -----------------------

            # keep a note of quality before change
            old_overall_n_grams_frequency_quality = self._overall_n_grams_frequency_quality

            # apply change now
            self._swap(cipher1, cipher2)

            if DEBUG:
                self._check_n_gram_frequency_quality()

            # did the quality improve ?
            if self._overall_n_grams_frequency_quality > old_overall_n_grams_frequency_quality:
                # yes : stop looping : we have improved
                return True

            # no improvement so undo
            self._swap(cipher1, cipher2)

            if not neighbours:
                return False

            if DEBUG:
                self._check_n_gram_frequency_quality()

    def _climb(self) -> None:
        """ climb : keeps going up until fails to do so """

        while True:

            # keeps climbing until fails to do so

            # up
            succeeded = self._go_up()
            if succeeded:
                if VERBOSE:
                    print(f"/", end='', flush=True)
            else:
                # slightly down
                succeeded = self._go_slightly_down()
                if succeeded:
                    if VERBOSE:
                        print(f"\\", end='', flush=True)
                else:
                    if VERBOSE:
                        print(f"-", flush=True)

                    print(f"Reached a peak at qual={self._overall_n_grams_frequency_quality}")
                    return

    def _reset_frequencies(self) -> None:
        """ reset_frequencies """

        assert CIPHER is not None
        assert DECRYPTER is not None
        assert NGRAMS is not None

        # n grams ==
        self._n_grams_frequency_quality_table.clear()
        # n_gram frequency quality table
        for n_gram in CIPHER.n_grams_set:
            # plain
            plain = DECRYPTER.decode_some(n_gram)
            # remembered
            self._n_grams_frequency_quality_table[n_gram] = NGRAMS.log_freq_table.get(plain, NGRAMS.worst_frequency) * CIPHER.n_grams_number_occurence_table[n_gram]

        # n_gram overall frequency quality of cipher
        # summed
        self._overall_n_grams_frequency_quality = sum(self._n_grams_frequency_quality_table.values())

        # simulated annealing
        self._temperature = K_TEMPERATURE_ZERO

    def make_tries(self) -> typing.Tuple[Evaluation, typing.Dict[str, str], float]:
        """ make tries : this includes  random generator and inner hill climb """

        assert DECRYPTER is not None
        assert DICTIONARY is not None
        assert CIPHER is not None

        # time before all climbs
        self._time_climbs_starts = time.time()

        # records best quality reached
        best_quality_reached: typing.Optional[Evaluation] = None

        # limit the number of climbs
        number_climbs_left = MAX_ATTACKER_CLIMBS

        while True:

            # pure random allocation
            initial_key = make_random_key()
            DECRYPTER.instantiate(initial_key)

            # reset frequency tables from new allocation
            self._reset_frequencies()

            # start a new session : climb as high as possible

            # actual climb
            self._climb()

            quality_ngrams_reached = self._overall_n_grams_frequency_quality
            quality_reached = Evaluation(quality_ngrams_reached)
            key_reached = DECRYPTER.allocation()

            # handle local best quality
            if best_quality_reached is None or quality_reached > best_quality_reached:

                best_quality_reached = quality_reached
                best_key_reached = key_reached

                # restart a complete climb from here (removed)
                number_climbs_left = MAX_ATTACKER_CLIMBS

                if IMPATIENT:
                    print(f"Putative solution below: ")
                    solution = Solution(quality_reached, key_reached)
                    solution.print_solution(sys.stdout)

            # stop at some point inner hill climb
            number_climbs_left -= 1
            if not number_climbs_left:

                # time after all climbs
                now = time.time()
                speed = self._n_operations / (now - self._time_climbs_starts)
                return best_quality_reached, best_key_reached, speed

    @property
    def overall_n_grams_frequency_quality(self) -> float:
        """ property """
        return self._overall_n_grams_frequency_quality


ATTACKER: typing.Optional[Attacker] = None


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--processes', required=True, help='how many processes to use')
    parser.add_argument('-i', '--ioc', required=True, help='input a file with index coincidence for language')
    parser.add_argument('-n', '--ngrams', required=True, help='input a file with frequency table for n_grams (n-letters)')
    parser.add_argument('-d', '--dictionary', required=True, help='input a file with frequency table for words (dictionary) to use')
    parser.add_argument('-L', '--limit', required=False, help='limit for the dictionary words to use')
    parser.add_argument('-l', '--letters', required=True, help='input a file with frequency table for letters')
    parser.add_argument('-c', '--cipher', required=True, help='cipher to attack')
    parser.add_argument('-o', '--output_solutions', required=False, help='file where to output successive solutions')
    args = parser.parse_args()

    n_processes = int(args.processes)
    print(f"INFORMATION: Using {n_processes} processes")

    ref_ioc_file = args.ioc
    if ref_ioc_file is not None:
        load_reference_coincidence_index(ref_ioc_file)

    letters_file = args.letters
    global LETTERS
    LETTERS = Letters(letters_file)
    #  print(LETTERS)

    ngrams_file = args.ngrams
    global NGRAMS
    NGRAMS = Ngrams(ngrams_file)
    #  print(NGRAMS)

    dictionary_file = args.dictionary
    limit = int(args.limit) if args.limit is not None else None
    global DICTIONARY
    DICTIONARY = Dictionary(dictionary_file, limit)
    #  print(DICTIONARY)

    cipher_file = args.cipher
    global CIPHER
    CIPHER = Cipher(cipher_file)
    #  print(f"Cipher='{CIPHER}'")

    global DECRYPTER
    DECRYPTER = Decrypter()

    global ATTACKER
    ATTACKER = Attacker()

    # file to best solution online
    output_solutions_file = args.output_solutions

    best_quality_reached: typing.Optional[Evaluation] = None

    # outer hill climb
    while True:

        # inner hill climb (includes random start key generator)
        quality_reached, key_reached, speed = ATTACKER.make_tries()

        # show new bucket
        print("=============================================")
        print(f"Yield a solution with quality={quality_reached} at speed={speed} swaps per sec")

        # if beaten global : update and show stuff
        if best_quality_reached is None or quality_reached > best_quality_reached or quality_reached == best_quality_reached:
            solution = Solution(quality_reached, key_reached)
            solution.print_solution(sys.stdout)
            if output_solutions_file is not None:
                with open(output_solutions_file, 'w') as file_handle:
                    solution.print_solution(file_handle)
            best_quality_reached = quality_reached


if __name__ == '__main__':

    sys.setrecursionlimit(RECURSION_LIMIT)

    # this if script too slow and profile it
    if PROFILE:
        PR = cProfile.Profile()
        PR.enable()

    # this to know how long it takes
    START = time.time()

    print("Press Ctrl+C to stop program")
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+C detected !")

    END = time.time()
    ELAPSED = END - START
    #  how long it took
    print(f"Time taken is {ELAPSED:2.2f}sec.")

    # stats
    if PROFILE:
        PR.disable()
        PS = pstats.Stats(PR)
        PS.strip_dirs()
        PS.sort_stats('time')
        PS.print_stats()  # uncomment to have profile stats

    sys.exit(0)
