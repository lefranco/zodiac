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
import itertools
import copy
import contextlib
import pprint
import secrets  # instead of random

import cProfile
import pstats

PROFILE = False
DEBUG = False
USE_OIC = False

RECURSION_LIMIT = 1500  # default is 1000 - this is only for when putting spaces when displaying clear text

ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]  # plain always lower case
EPSILON_NO_OCCURENCES = 1e-99  # zero has - infinite as log, must be << 1
EPSILON_DELTA_FLOAT = 0.000001  # to compare floats

MAX_SUBSTITUTION_STUFFING = 10
MAX_BUCKET_SIZE = 99   # keep it to two digit

K_CIPHER_DIFFICULTY = 50.
MIN_ATTACKER_CLIMBS = 10
MAX_ATTACKER_CLIMBS = 100

REF_IOC = 0.


def load_reference_coincidence_index(filename: str) -> None:
    """ Loads IOC from file """
    with open(filename) as filepointer:
        for line in filepointer:
            line = line.rstrip('\n')
            global REF_IOC
            REF_IOC = float(line)
            print(f"INFORMATION  :reference IOC is {REF_IOC}")
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

        raw_frequency_table: typing.Dict[str, int] = dict()
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                n_gram_read, frequency_str = line.split()
                n_gram = n_gram_read.lower()
                if self._size:
                    assert len(n_gram) == self._size, "Problem with ngram file content"
                else:
                    self._size = len(n_gram)
                    print(f"INFORMATION: Using N-Grams with N={self._size}")
                frequency = int(frequency_str)
                raw_frequency_table[n_gram] = frequency

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

        best_words = segment_rec(plain)
        return words_probability(best_words), best_words

    def __str__(self) -> str:
        """ for debug """
        return pprint.pformat(self._log_frequency_table)


DICTIONARY: typing.Optional[Dictionary] = None


class Cipher:
    """ A cipher : basically a string """

    def __init__(self, filename: str, substitution_mode: bool) -> None:

        assert NGRAMS is not None

        # the string (as a list) read from cipher file
        self._content: typing.List[str] = list()
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                for word in line.split():
                    for code in word:

                        # substituion mode : check in alphabet, store as upper case
                        if substitution_mode:
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

        # a table where how many times code appear in cipher
        self._codes_number_occurence_table = collections.Counter(self._cipher_str)

        # a table where how many times n_grams appear in cipher
        self._n_grams_number_occurence_table = collections.Counter(cipher_n_grams)

        # a table from a cipher to n_grams that contain it
        self._n_grams_localization_table: typing.Dict[str, typing.List[str]] = collections.defaultdict(list)
        for n_gram in self._n_grams_set:
            for code in n_gram:
                self._n_grams_localization_table[code].append(n_gram)

    def difficulty(self) -> int:
        """ climb_difficulty """
        print(f"INFORMATION: We have a cipher with {len(self._cipher_codes)} different codes and a length of {len(self._content)}")
        difficulty_from_codes = (len(self._cipher_codes) / len(ALPHABET)) ** 2
        easiness_from_size = math.sqrt(len(self._content) / len(self._cipher_codes))
        res = int(K_CIPHER_DIFFICULTY * difficulty_from_codes / easiness_from_size)
        return res

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
    def codes_number_occurence_table(self) -> typing.Counter[str]:
        """ property """
        return self._codes_number_occurence_table

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
        return copy.copy(self._table)

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
        #  assert plain1 != plain2, "Internal error"

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


class Bucket:
    """ A bucket : says how many ciphers could be allocated to every alphabet letter """

    def __init__(self, substitution_mode: bool, hint_file: typing.Optional[str]) -> None:

        assert CIPHER is not None

        # set of buckets tried to avoid repeat (in tuple form)
        self._bucket_tried: typing.Set[typing.Tuple[int, ...]] = set()

        if substitution_mode:

            assert hint_file is None, "There cannot be a hint file in substitution mode"
            self._table = {ll: 1 for ll in sorted(ALPHABET, key=lambda ll: LETTERS.freq_table[ll], reverse=True)}  # type: ignore
            return

        if hint_file is not None:

            with open(hint_file) as filepointer:
                for num, line in enumerate(filepointer):
                    line = line.rstrip('\n')
                    if num in [0, 3]:
                        assert line == '-' * len(ALPHABET), f"Incorrect hint file line {num+1}"
                    elif num == 1:
                        assert line == ''.join(ALPHABET), f"Incorrect hint file line {num+1}"
                    elif num == 2:
                        self._table = {ll: int(line[n]) if line[n] != ' ' else 0 for n, ll in enumerate(ALPHABET)}
            return

        # standard

        #  how many different codes
        number_codes = len(CIPHER.cipher_codes)
        self._table = {ll: 0 for ll in sorted(ALPHABET, key=lambda ll: LETTERS.freq_table[ll], reverse=True)}  # type: ignore

        while True:

            # criterion is deficit : how many I should have minus how many I have
            chosen = max(ALPHABET, key=lambda ll: LETTERS.freq_table[ll] * number_codes - self._table[ll])  # type: ignore
            self._table[chosen] += 1

            if sum(self._table.values()) == number_codes:
                break

    def _do_fake_swap(self, decremented: str, incremented: str) -> None:
        """ swap letters in allocator """

        # just a little check
        assert incremented != decremented, "Internal error"

        assert self._table[decremented], "Internal error"
        self._table[decremented] -= 1
        self._table[incremented] += 1
        assert self._table[incremented] <= MAX_BUCKET_SIZE, f"Cannot handle buckets with more than {MAX_BUCKET_SIZE} capacity"

    def _evaluate_fake_swap(self, decremented: str, incremented: str) -> float:
        """ evaluate swap letters in allocator """

        assert LETTERS is not None

        table_changed = copy.copy(self._table)
        assert table_changed[decremented]
        table_changed[decremented] -= 1
        table_changed[incremented] += 1
        bucket_capture_changed = tuple(table_changed.values())

        sum_occurences = sum(bucket_capture_changed)
        frequencies = {ll: bucket_capture_changed[n] / sum_occurences for n, ll in enumerate(ALPHABET)}
        return - sum([abs(frequencies[ll] - LETTERS.freq_table[ll]) for ll in frequencies])

    def find_apply_fake_swap(self) -> bool:
        """ find a new bucket swap return success """

        # all possible pseudo swaps
        possibles_swaps = [(d, i) for (d, i) in itertools.permutations(self._table, 2) if self._table[d]]

        # sorted by quality
        possibles_swaps_sorted = sorted(possibles_swaps, key=lambda s: self._evaluate_fake_swap(s[0], s[1]), reverse=True)

        while True:

            # take first best swap
            decremented, incremented = possibles_swaps_sorted.pop(0)

            # is it new ?
            self._do_fake_swap(decremented, incremented)
            new_bucket_capture = tuple(self._table.values())
            if new_bucket_capture not in self._bucket_tried:
                # show
                print(f"decremented {decremented} / incremented {incremented}")
                return True

            # undo bucket swap because already done
            self._do_fake_swap(incremented, decremented)  # pylint: disable=arguments-out-of-order

            # is there any change still possible ?
            if not possibles_swaps_sorted:
                print("Cannot do any more bucket change")
                return False

    def remember(self) -> None:
        """ remember """
        bucket_capture = tuple(self._table.values())
        self._bucket_tried.add(bucket_capture)

    def print_repartition(self, file_handle: typing.TextIO) -> None:
        """ print_repartition """

        with contextlib.redirect_stdout(file_handle):
            print("." * len(ALPHABET))
            print(''.join(ALPHABET))
            for letter in ALPHABET:
                number = self._table[letter]
                if number:
                    print(number % 10, end='')
                else:
                    print(' ', end='')
            print()
            for letter in ALPHABET:
                number = self._table[letter]
                if number // 10:
                    print(number // 10, end='')
                else:
                    print(' ', end='')
            print()
            print("." * len(ALPHABET))

    @property
    def table(self) -> typing.Dict[str, int]:
        """ property """
        return self._table


BUCKET: typing.Optional[Bucket] = None


class Allocator:
    """ Alocator : Makes initial random allocation """

    def __init__(self, substitution_mode: bool) -> None:
        self._substitution_mode = substitution_mode

    def make_allocation(self) -> typing.Dict[str, str]:
        """ Makes a random allocation to start with """

        assert CIPHER is not None
        assert BUCKET is not None

        bucket_copy = copy.copy(BUCKET.table)

        allocation: typing.Dict[str, str] = dict()

        for cipher in CIPHER.cipher_codes:
            letter_selected = secrets.choice([ll for ll in bucket_copy if bucket_copy[ll]])
            allocation[cipher] = letter_selected
            bucket_copy[letter_selected] -= 1

        if self._substitution_mode and len(allocation) < len(ALPHABET):
            stuffing = sorted(set(ALPHABET) - set(allocation.values()))
            assert len(stuffing) <= MAX_SUBSTITUTION_STUFFING, "Too few different characters in substitution cipher"
            num = 0
            while True:
                letter_selected = secrets.choice(stuffing)
                stuffing.remove(letter_selected)
                allocation[f'{num}'] = letter_selected
                bucket_copy[letter_selected] -= 1
                num += 1
                if not stuffing:
                    break

        assert all([bucket_copy[ll] == 0 for ll in bucket_copy]), "Internal error"

        return allocation


ALLOCATOR: typing.Optional[Allocator] = None
CLIMBS_STARTS = 0.

class Evaluation:
    """ Evaluation """

    def __init__(self, coincidence_index_quality: float, n_grams_frequency_quality: float) -> None:

        self._coincidence_index_quality = coincidence_index_quality
        self._n_grams_frequency_quality = n_grams_frequency_quality

    @property
    def coincidence_index_quality(self) -> float:
        """ property """
        return self._coincidence_index_quality

    @property
    def n_grams_frequency_quality(self) -> float:
        """ property """
        return self._n_grams_frequency_quality

    def __gt__(self, other: 'Evaluation') -> bool:

        # IOC (using it alone lowers results a lot)
        # but it should be prefered at begining of hill climbing when quality is bad

        if USE_OIC:
            return self._coincidence_index_quality > other.coincidence_index_quality

        return self._n_grams_frequency_quality > other.n_grams_frequency_quality

    def __str__(self) -> str:
        return f"delta IOC={self._coincidence_index_quality} ngram qual={self._n_grams_frequency_quality}"


class Solution:
    """ A solution """

    def __init__(self, allocation: typing.Dict[str, str], quality: Evaluation) -> None:

        self._allocation = copy.copy(allocation)
        self._quality = quality

    def print_solution(self, file_handle: typing.TextIO) -> None:
        """ show print_solution """

        assert DICTIONARY is not None

        # get speed
        now = time.time()
        speed = N_OPERATIONS / (now - CLIMBS_STARTS)

        # get dictionary quality of result
        my_decrypter = Decrypter()
        my_decrypter.instantiate(self._allocation)
        plain = my_decrypter.apply()
        dictionary_quality, selected_words = DICTIONARY.extracted_words(plain)

        # print stuff
        print(f"Speed is {speed} checks per sec", file=file_handle)
        print("=" * 50, file=file_handle)
        print(' '.join(selected_words), file=file_handle)
        print("=" * 50, file=file_handle)
        print(f"dictionary_quality={dictionary_quality}", file=file_handle)
        print(f"quality=[{self._quality}]", file=file_handle)
        my_decrypter.print_key(file_handle)


BEST_QUALITY_REACHED: typing.Optional[Evaluation] = None
N_OPERATIONS = 0


class Attacker:
    """ Attacker """

    def __init__(self, solution_filename: typing.Optional[str], substitution_mode: bool) -> None:

        assert CIPHER is not None

        # a table for remembering frequencies
        self._n_grams_frequency_quality_table: typing.Dict[str, float] = dict()

        # a table for remembering nb occurences plain
        self._nb_occ_plain: typing.Dict[str, int] = collections.defaultdict(int)

        self._overall_n_grams_frequency_quality = 0.
        self._overall_coincidence_index_quality = 0

        if substitution_mode:
            self._number_climbs = MAX_ATTACKER_CLIMBS
        else:
            # number of climbs = cipher difficulty
            self._number_climbs = CIPHER.difficulty()
            self._number_climbs = min(MAX_ATTACKER_CLIMBS, self._number_climbs)
            self._number_climbs = max(MIN_ATTACKER_CLIMBS, self._number_climbs)

        print(f"INFORMATION: Inner hill climb will use max={self._number_climbs}")

        self._solution_filename = solution_filename

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

    def _check_index_coincidence_quality(self) -> None:
        """ Evaluates quality from index coincidence DEBUG """

        assert DEBUG
        assert USE_OIC

        assert DECRYPTER is not None

        # debug check
        plain = DECRYPTER.apply()
        assert all([plain.count(ll) == self._nb_occ_plain[ll] for ll in set(plain)])

        qcheck = sum([self._nb_occ_plain[p] * (self._nb_occ_plain[p] - 1) for p in self._nb_occ_plain])

        assert qcheck == self._overall_coincidence_index_quality, "Debug mode detected an error for IOC"

    def _swap(self, cipher1: str, cipher2: str) -> None:
        """ swap: this is where most CPU time is spent in the program """

        assert NGRAMS is not None
        assert CIPHER is not None
        assert DECRYPTER is not None

        if USE_OIC:
            # IOC obliterated
            for cipher in cipher1, cipher2:
                plain = DECRYPTER.decode_some(cipher)
                self._overall_coincidence_index_quality -= self._nb_occ_plain[plain] * (self._nb_occ_plain[plain] - 1)
                self._nb_occ_plain[plain] -= CIPHER.codes_number_occurence_table[cipher]

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

        if USE_OIC:
            # new IOC
            for cipher in cipher1, cipher2:
                plain = DECRYPTER.decode_some(cipher)
                self._nb_occ_plain[plain] += CIPHER.codes_number_occurence_table[cipher]
                self._overall_coincidence_index_quality += self._nb_occ_plain[plain] * (self._nb_occ_plain[plain] - 1)

    def _go_up(self) -> bool:
        """ go up : try to improve things... """

        assert CIPHER is not None
        assert DECRYPTER is not None

        neighbours = {(c1, c2) for (p1, p2) in itertools.combinations(DECRYPTER.allocated, 2) for c1 in DECRYPTER.reverse_table[p1] for c2 in DECRYPTER.reverse_table[p2]}

        while True:

            # take a random neighbour
            neighbour = secrets.choice(list(neighbours))
            neighbours.remove(neighbour)
            cipher1, cipher2 = neighbour

            # -----------------------
            #  does the change improve things ?
            # -----------------------

            # keep a note of quality before change
            old_overall_n_grams_frequency_quality = self._overall_n_grams_frequency_quality
            old_overall_coincidence_index_quality = self._overall_coincidence_index_quality

            # apply change now
            self._swap(cipher1, cipher2)

            # debug purpose only
            global N_OPERATIONS
            N_OPERATIONS += 1

            if DEBUG:
                self._check_n_gram_frequency_quality()
                if USE_OIC:
                    self._check_index_coincidence_quality()

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
                if USE_OIC:
                    self._check_index_coincidence_quality()

            # restore value
            self._overall_n_grams_frequency_quality = old_overall_n_grams_frequency_quality
            self._overall_coincidence_index_quality = old_overall_coincidence_index_quality

    def _climb(self) -> None:
        """ climb : keeps going up until fails to do so """

        while True:

            # keeps climbing until fails to do so
            succeeded = self._go_up()
            if not succeeded:
                print("-", flush=True)
                return
            print("/", end='', flush=True)

    def _reset_frequencies(self) -> None:
        """ reset_frequencies """

        assert CIPHER is not None
        assert DECRYPTER is not None
        assert NGRAMS is not None

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

        # IOC
        self._overall_coincidence_index_quality = 0.
        if USE_OIC:
            self._nb_occ_plain.clear()
            for cipher in CIPHER.cipher_codes:
                plain = DECRYPTER.decode_some(cipher)
                self._nb_occ_plain[plain] += CIPHER.codes_number_occurence_table[cipher]
            self._overall_coincidence_index_quality = sum([self._nb_occ_plain[p] * (self._nb_occ_plain[p] - 1) for p in self._nb_occ_plain])

    def make_tries(self) -> Evaluation:
        """ make tries : this includes  random generator and inner hill climb """

        assert ALLOCATOR is not None
        assert DECRYPTER is not None
        assert DICTIONARY is not None
        assert CIPHER is not None

        # records best quality reached
        best_quality_reached: typing.Optional[Evaluation] = None

        # limit the number of climbs
        number_climbs_left = self._number_climbs
        while True:

            # random allocator
            initial_allocation = ALLOCATOR.make_allocation()
            DECRYPTER.instantiate(initial_allocation)

            # reset frequency tables from new allocation
            self._reset_frequencies()

            # start a new session : climb as high as possible
            self._climb()

            quality_coincidence_reached = - abs(len(ALPHABET) * self._overall_coincidence_index_quality / (len(CIPHER.cipher_str) * (len(CIPHER.cipher_str) - 1)) - REF_IOC)
            quality_ngrams_reached = self._overall_n_grams_frequency_quality
            quality_reached = Evaluation(quality_coincidence_reached, quality_ngrams_reached)

            # handle local best quality
            if best_quality_reached is None or quality_reached > best_quality_reached:

                # beaten local, show stuff if also beaten global
                if BEST_QUALITY_REACHED is None or quality_reached > BEST_QUALITY_REACHED:
                    allocation = DECRYPTER.allocation()
                    solution = Solution(allocation, quality_reached)
                    solution.print_solution(sys.stdout)
                    if self._solution_filename is not None:
                        with open(self._solution_filename, 'w') as file_handle:
                            solution.print_solution(file_handle)

                best_quality_reached = quality_reached

                # restart a complete climb from here
                number_climbs_left = self._number_climbs

            # stop at some point inner hill climb
            number_climbs_left -= 1
            if not number_climbs_left:
                return best_quality_reached

    @property
    def overall_n_grams_frequency_quality(self) -> float:
        """ property """
        return self._overall_n_grams_frequency_quality


ATTACKER: typing.Optional[Attacker] = None


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ic', required=True, help='input a file with index coincidence for language')
    parser.add_argument('-n', '--ngrams', required=True, help='input a file with frequency table for n_grams (n-letters)')
    parser.add_argument('-d', '--dictionary', required=True, help='input a file with frequency table for words (dictionary) to use')
    parser.add_argument('-L', '--limit', required=False, help='limit for the dictionary words to use')
    parser.add_argument('-l', '--letters', required=True, help='input a file with frequency table for letters')
    parser.add_argument('-c', '--cipher', required=True, help='cipher to attack')
    parser.add_argument('-o', '--output', required=False, help='file where to output successive solutions')
    parser.add_argument('-H', '--hint_file', required=False, help='file with hint (sizes of buckets) in cipher')
    parser.add_argument('-s', '--substitution_mode', required=False, help='cipher is simple substitution (not homophonic)', action='store_true')
    args = parser.parse_args()

    ref_ic_file = args.ic
    load_reference_coincidence_index(ref_ic_file)

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

    substitution_mode = args.substitution_mode
    cipher_file = args.cipher
    global CIPHER
    CIPHER = Cipher(cipher_file, substitution_mode)
    #  print(f"Cipher='{CIPHER}'")

    global DECRYPTER
    DECRYPTER = Decrypter()

    hint_file = args.hint_file
    global BUCKET
    BUCKET = Bucket(substitution_mode, hint_file)
    print("Initial Bucket:")
    BUCKET.print_repartition(sys.stdout)

    global ALLOCATOR
    ALLOCATOR = Allocator(substitution_mode)
    #  print(f"Allocator='{ALLOCATOR}'")

    output_file = args.output
    global ATTACKER
    ATTACKER = Attacker(output_file, substitution_mode)

    global CLIMBS_STARTS
    CLIMBS_STARTS = time.time()

    # outer hill climb
    while True:

        # inner hill climb (includes random allocator)
        quality_reached = ATTACKER.make_tries()

        # actually these are test modes
        if substitution_mode or hint_file is not None:
            break

        # if this inner clib hill improved things, record it
        global BEST_QUALITY_REACHED
        if BEST_QUALITY_REACHED is None or quality_reached > BEST_QUALITY_REACHED:
            BEST_QUALITY_REACHED = quality_reached

        # remember this bucket as done
        BUCKET.remember()

        # change bucket if still possible
        status = BUCKET.find_apply_fake_swap()
        if not status:
            break

        # show new bucket
        print("=============================================")
        print("New Bucket:")
        BUCKET.print_repartition(sys.stdout)


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
    print(f"Time taken is {ELAPSED:2.2f}sec.")

    # stats
    if PROFILE:
        PR.disable()
        PS = pstats.Stats(PR)
        PS.strip_dirs()
        PS.sort_stats('time')
        PS.print_stats()  # uncomment to have profile stats

    sys.exit(0)
