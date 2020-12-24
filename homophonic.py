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
import copy
import pprint

import cProfile
import pstats

PROFILE = False
DEBUG = True

ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]

NGRAMS = 4

EPSILON_NO_OCCURENCES = 0.1  # zero has - infinite as log, must be << 1
EPSILON_PROBA = 1 / 1000  # to make sure we can give up searching
EPSILON_DELTA_FLOAT = 0.000001  # to compare floats


def load_frequency_table(filename: str) -> typing.Dict[str, float]:
    """ load_frequency_table """

    before = time.time()

    raw_frequency_table: typing.Dict[str, int] = dict()
    with open(filename) as filepointer:
        for line in filepointer:
            line = line.rstrip('\n')
            quadgram_read, frequency_str = line.split()
            quadgram = quadgram_read.lower()
            frequency = int(frequency_str)
            raw_frequency_table[quadgram] = frequency

    coverage = (len(raw_frequency_table) / (len(ALPHABET) ** NGRAMS)) * 100
    print(f"Frequency tables covers {coverage:.2f}% of possibilities")

    sum_occurences = sum(raw_frequency_table.values())

    # for normal values
    frequency_table = {q: math.log10(raw_frequency_table[q] / sum_occurences) for q in raw_frequency_table}

    # complete for absent values
    def_log_value = math.log10(EPSILON_NO_OCCURENCES / sum_occurences)
    frequency_table.update({''.join(letters): def_log_value for letters in itertools.product(ALPHABET, repeat=NGRAMS) if ''.join(letters) not in frequency_table})

    after = time.time()
    elapsed = after - before
    print(f"N-Gram frequency file '{filename}' loaded in {elapsed:2.2f} seconds")

    return frequency_table


NGRAMS_FREQUENCY_TABLE: typing.Dict[str, float] = dict()


class Node:
    """ Node """

    def __init__(self) -> None:
        self._word: typing.Optional[str] = None
        self._log_frequency: typing.Optional[float] = None
        self._children: typing.Dict[str, 'Node'] = collections.defaultdict(Node)

    def add_word(self, word: str, rest_word: str, log_frequency: float) -> None:
        """ add_word """
        if not rest_word:
            self._word = word
            self._log_frequency = log_frequency
            return
        first, rest = rest_word[0], rest_word[1:]
        node = self._children[first]
        node.add_word(word, rest, log_frequency)

    @property
    def word(self) -> typing.Optional[str]:
        """ property """
        return self._word

    @property
    def log_frequency(self) -> typing.Optional[float]:
        """ property """
        return self._log_frequency

    @property
    def children(self) -> typing.Dict[str, 'Node']:
        """ property """
        return self._children


def debug(num: int, node: Node) -> None:
    """ debug : displaying recursively a trie """
    if node.word:
        print(f"{' '*num}node.word={node.word} node.log_frequency={node.log_frequency}")
    for char, sub_node in node.children.items():
        print(f"{' '*num}{char} : ")
        debug(num + 1, sub_node)


class Trie:
    """ Trie """

    def __init__(self) -> None:
        self._root = Node()

    def add_word(self, word: str, log_frequency: float) -> None:
        """ add_word """
        self._root.add_word(word, word, log_frequency)

    def possible_words(self, clear: str) -> typing.Generator[str, None, None]:
        """ generator of all possible words in the clear string """

        node = self._root
        rest = clear
        while True:
            if not rest:
                break
            first, rest = rest[0], rest[1:]
            if first not in node.children:
                break
            node = node.children[first]
            if node.word is not None:
                assert node.log_frequency is not None
                yield node.word


class Dictionary:
    """ Stores the list of word. Say how many words in tempted clear """

    def __init__(self, filename: str, limit: typing.Optional[int]) -> None:

        before = time.time()

        self._log_frequency_table: typing.Dict[str, float] = dict()
        self._trie = Trie()

        temp_frequency_table: typing.Dict[str, int] = dict()

        # pass one : read the file
        line_num = 1
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                word, frequency_str = line.split()
                word = word.lower()

                assert not [ll for ll in word if ll not in ALPHABET], f"ERROR : bad word found in dictionary line {line_num} : <{word}>"

                if word in temp_frequency_table:
                    print(f"Warning : duplicated word '{word}' for dictionary line {line_num}")
                    continue

                frequency = int(frequency_str)
                temp_frequency_table[word] = frequency
                line_num += 1

                if limit is not None and len(temp_frequency_table) == limit:
                    break

        # pass two : enter data
        sum_frequencies = sum(temp_frequency_table.values())
        for word, frequency in temp_frequency_table.items():
            log_frequency = math.log(frequency / sum_frequencies)
            self._log_frequency_table[word] = log_frequency
            # added in trie of words
            self._trie.add_word(word, log_frequency)

        self._worst_frequency = min(self._log_frequency_table.values())

        after = time.time()
        elapsed = after - before
        print(f"Word list file '{filename}' loaded in {elapsed:2.2f} seconds")

    def detected_words(self, clear: str) -> typing.Dict[int, typing.List[str]]:
        """ parse clear to find a list of consecutive words  """

        # find all the words present
        detected_words: typing.Dict[int, typing.List[str]] = collections.defaultdict(list)
        start = 0
        while True:
            # rank is not used
            # only longest word is taken
            for word in self._trie.possible_words(clear[start:]):
                detected_words[start].append(word)
            start += 1
            if start >= len(clear):
                break

        return detected_words

    def quality_from_dictionary(self, clear: str) -> typing.Tuple[float, typing.Dict[int, str]]:
        """ Tells the  of (more or less) clear text from the dictionary (the trie mainly)  """

        def heuristic(word: str) -> float:
            """ heuristic for a decision of taking which word(s) """
            return self._log_frequency_table[word] / len(word)

        readables = self.detected_words(clear)

        selected_words: typing.Dict[int, str] = dict()
        position = 0
        while True:

            # finished
            if position == len(clear):
                break

            # no word starting from here
            if not readables[position]:
                position += 1
                continue

            # which word do we choose from here ?
            # we really prefer longer words

            #  debug
            #  for w in readables[position]:
                #  print(f"{w} --> {heuristic(w)}")

            selected_word = max(readables[position], key=heuristic)
            selected_words[position] = selected_word
            position += len(selected_word)

        score_from_selected = sum(map(lambda w: self._log_frequency_table[w], selected_words.values()))

        num_undiciphered = len(clear) - sum(map(len, selected_words.values()))
        score_from_undiciphered = num_undiciphered * self._worst_frequency

        #  print(f"{score_from_undiciphered=} {score_from_undiciphered=} ({num_undiciphered} {self._worst_frequency=})")
        quality = score_from_selected + score_from_undiciphered
        return quality, selected_words

    def __str__(self) -> str:
        """ for debug """
        return pprint.pformat(self._log_frequency_table)


DICTIONARY: typing.Optional[Dictionary] = None


class Cipher:
    """ A cipher : basically a string """

    def __init__(self, filename: str, debug_mode: bool) -> None:

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
                                print(f"In debug mode igored  '{code}' from pseudo cipher")
                                continue
                        self._content.append(code)

        # the different codes in cipher
        self._cipher_codes = list(set(self._content))

        # a table where every ngram appear (actually only len is used - how many times it appears)
        self._cipher_str = ''.join(self._content)
        self._quadgram_occurence_table: typing.Dict[str, typing.List[int]] = collections.defaultdict(list)
        for start in range(len(self._cipher_str) - (NGRAMS - 1)):
            quadgram = self._cipher_str[start: start + NGRAMS]
            self._quadgram_occurence_table[quadgram].append(start)

        # a table to speed up for when changing a ngram finding which others are impacted
        self._quadgram_interaction_table: typing.Dict[str, typing.Dict[int, typing.Set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))
        for quadgram1, quadgram2 in itertools.permutations(self._quadgram_occurence_table, 2):
            for position in range(NGRAMS):
                if quadgram1[position] in quadgram2:
                    self._quadgram_interaction_table[quadgram1][position].add(quadgram2)

    def display_result(self, selected_words: typing.Dict[int, str]) -> None:
        """ Display in a clear form the result of decoding """
        pos = 0
        junk_before = False
        while True:
            assert pos <= len(self._cipher_str)
            if pos == len(self._cipher_str):
                break
            if pos in selected_words:
                if junk_before:
                    print(' ', end='')
                word = selected_words[pos]
                print(word, end=' ')
                pos += len(word)
                junk_before = False
            else:
                print('?', end='')
                pos += 1
                junk_before = True
        print()

    @property
    def cipher_codes(self) -> typing.List[str]:
        """ property """
        return self._cipher_codes

    @property
    def cipher_str(self) -> str:
        """ property """
        return self._cipher_str

    @property
    def quadgram_occurence_table(self) -> typing.Dict[str, typing.List[int]]:
        """ property """
        return self._quadgram_occurence_table

    @property
    def quadgram_interaction_table(self) -> typing.Dict[str, typing.Dict[int, typing.Set[str]]]:
        """ property """
        return self._quadgram_interaction_table

    def __str__(self) -> str:
        return self._cipher_str


CIPHER: typing.Optional[Cipher] = None


class Decrypter:
    """ A decrypter : basically a dictionary """

    def __init__(self, cipher_codes: typing.List[str]) -> None:
        self._table = {c: '' for c in cipher_codes}

    def decode(self, code: str) -> str:
        """ decode """
        return self._table[code] if code in self._table else '?'

    def install(self, random_allocation: typing.Dict[str, str]) -> None:
        """ install an initial table """
        for code, clear in random_allocation.items():
            self._table[code] = clear

    def change(self, cipher_quadgram: str, new_clear_quadgram: str) -> None:
        """ change the table """
        for code, clear in zip(cipher_quadgram, new_clear_quadgram):
            self._table[code] = clear

    def apply(self, cipher: str) -> str:
        """ apply the table (get a new clear from a cipher) """
        return ''.join([self.decode(c) for c in cipher])

    def __eq__(self, other: object) -> bool:
        """ for debug """
        if not isinstance(other, Decrypter):
            return NotImplemented
        return self._table == other._table

    def __str__(self) -> str:
        return pprint.pformat(self._table)


DECRYPTER: typing.Optional[Decrypter] = None


class Attacker:
    """ Attacker """

    def __init__(self, debug_mode: bool) -> None:

        assert CIPHER is not None
        assert DECRYPTER is not None
        assert DICTIONARY is not None

        # make initial random allocation
        if debug_mode:
            allocation = {k: k for k in CIPHER.cipher_codes}
        else:
            allocation = {k: random.choice(ALPHABET) for k in CIPHER.cipher_codes}

        # put it in crypter
        DECRYPTER.install(allocation)

        # make table linking clear and cipher quadgrams
        self._quadgram_allocation_table: typing.Dict[str, str] = dict()
        self._reverse_quadgram_allocation_table: typing.Dict[str, typing.Set[str]] = collections.defaultdict(set)
        for quadgram_from_cipher in CIPHER.quadgram_occurence_table:
            quadgram_clear = DECRYPTER.apply(quadgram_from_cipher)
            self._quadgram_allocation_table[quadgram_from_cipher] = quadgram_clear
            # reverse
            self._reverse_quadgram_allocation_table[quadgram_clear].add(quadgram_from_cipher)

        # quadgram frequency quality
        # declare
        self._quadgram_frequency_quality = 0.
        # evaluate
        self._set_quadgram_frequency_quality()

        # dictionary quality
        clear = DECRYPTER.apply(CIPHER.cipher_str)
        new_dict_quality, _ = DICTIONARY.quality_from_dictionary(clear)
        self._dictionary_quality = new_dict_quality

        if DEBUG:
            self._check_sanity()

    @staticmethod
    def _accept(cipher_quadgram: str, clear_quadgram: str) -> bool:
        """ signature of clear must be same as signature of cipher """
        assert len(cipher_quadgram) == len(clear_quadgram)
        assert len(cipher_quadgram) == NGRAMS
        return not any([cipher_quadgram[p1] == cipher_quadgram[p2] and clear_quadgram[p1] != clear_quadgram[p2] for p1, p2 in itertools.combinations(range(NGRAMS), 2)])

    def _check_sanity(self) -> None:
        """ _check_sanity (for debug purpose) """

        assert DECRYPTER is not None

        assert DEBUG

        # check tables are coherent between themselves
        for cipher_quad, clear_quad in self._quadgram_allocation_table.items():
            assert cipher_quad in self._reverse_quadgram_allocation_table[clear_quad]

        for clear_quad, cipher_quads in self._reverse_quadgram_allocation_table.items():
            for cipher_quad in cipher_quads:
                assert self._quadgram_allocation_table[cipher_quad] == clear_quad

        # check tables are coherent with table in cipher
        for cipher_quad, clear_quad in self._quadgram_allocation_table.items():
            assert DECRYPTER.apply(cipher_quad) == clear_quad

    def _change(self, cipher_quadgram: str, quadgram_killed: str, quadgram_replacing: str, position: int) -> None:
        """ change : we replace for 'cipher_quadgram' corresponding 'quadgram_killed' by 'quadgram_replacing' - just one letter change at position 'position' """

        assert CIPHER is not None
        assert DECRYPTER is not None

        if DEBUG:
            assert quadgram_killed[:position] == quadgram_replacing[:position]
            assert quadgram_killed[position + 1:] == quadgram_replacing[position + 1:]
            assert quadgram_killed[position] != quadgram_replacing[position]

        # apply change
        previous_there = self._quadgram_allocation_table[cipher_quadgram]
        assert previous_there == quadgram_killed
        self._reverse_quadgram_allocation_table[previous_there].remove(cipher_quadgram)
        if not self._reverse_quadgram_allocation_table[previous_there]:
            del self._reverse_quadgram_allocation_table[previous_there]

        self._quadgram_allocation_table[cipher_quadgram] = quadgram_replacing

        # reverse
        self._reverse_quadgram_allocation_table[quadgram_replacing].add(cipher_quadgram)

        # and in conflicting
        for impacted_cipher_quadgram in CIPHER.quadgram_interaction_table[cipher_quadgram][position]:
            impacted_quadgram_replacing = DECRYPTER.apply(impacted_cipher_quadgram)
            assert len(impacted_quadgram_replacing) == NGRAMS
            impacted_previous_there = self._quadgram_allocation_table[impacted_cipher_quadgram]
            self._reverse_quadgram_allocation_table[impacted_previous_there].remove(impacted_cipher_quadgram)
            if not self._reverse_quadgram_allocation_table[impacted_previous_there]:
                del self._reverse_quadgram_allocation_table[impacted_previous_there]

            self._quadgram_allocation_table[impacted_cipher_quadgram] = impacted_quadgram_replacing

            # reverse
            self._reverse_quadgram_allocation_table[impacted_quadgram_replacing].add(impacted_cipher_quadgram)

    def _set_quadgram_frequency_quality(self) -> None:
        """ Evaluates quality from quadgram frequency """

        assert CIPHER is not None
        assert DECRYPTER is not None

        self._quadgram_frequency_quality = sum([len(CIPHER.quadgram_occurence_table[q]) * NGRAMS_FREQUENCY_TABLE[self._quadgram_allocation_table[q]] for q in self._quadgram_allocation_table])

        if DEBUG:
            self._check_sanity()
            # debug check
            qcheck = 0.
            clear = DECRYPTER.apply(CIPHER.cipher_str)
            for position in range(len(clear) - NGRAMS + 1):
                quadgram = clear[position: position + NGRAMS]
                qcheck += NGRAMS_FREQUENCY_TABLE[quadgram]
            assert abs(qcheck - self._quadgram_frequency_quality) < EPSILON_DELTA_FLOAT

    def _find_hold(self) -> typing.Optional[typing.Tuple[str, str, str, int]]:

        """ Find a  holds that leads to a possible change from this situation """

        assert DECRYPTER is not None
        assert CIPHER is not None

        # possibilities are number of quadgrams x (size of quadgram - 1)
        number = len(self._reverse_quadgram_allocation_table) * (NGRAMS - 1)
        attempts_find_change_left = int(math.log(EPSILON_PROBA) / math.log((number - 1) / number))

        while True:

            # randomly select a 'quadgram_killed' in the table
            quadgram_killed = random.choice(list(self._quadgram_allocation_table.values()))

            # randomly select a 'cipher_quadgram' corresponding (usually there is only one)
            cipher_quadgram = random.choice(list(self._reverse_quadgram_allocation_table[quadgram_killed]))

            # randomly select a letter in the quadgram
            position = random.randint(0, NGRAMS - 1)

            # randomly select a new letter
            old_letter = quadgram_killed[position]
            letters = set(ALPHABET) - set([old_letter])
            letter = random.choice(list(letters))

            quadgram_replacing = quadgram_killed[:position] + letter + quadgram_killed[position + 1:]
            assert len(quadgram_replacing) == len(quadgram_killed)

            # must respect pattern
            if self._accept(cipher_quadgram, quadgram_replacing):
                return cipher_quadgram, quadgram_killed, quadgram_replacing, position

            attempts_find_change_left -= 1
            if attempts_find_change_left == 0:
                #  print("Giving up... difficult find a change...")
                return None

    def _climb(self) -> bool:
        """ climb : try to improve things... """

        assert DECRYPTER is not None

        # possibilities are 26 * 25 (choice of two letters in alphabet)
        number = len(ALPHABET) * (len(ALPHABET) - 1)
        attempts_find_improvement_left = int(math.log(EPSILON_PROBA) / math.log((number - 1) / number))
        while True:

            # -----------------------
            # find a possible change
            # -----------------------

            hold = self._find_hold()
            if hold is None:
                return False

            cipher_quadgram, quadgram_killed, quadgram_replacing, position = hold

            # -----------------------
            #  does the change improve things ?
            # -----------------------

            # for debug purpose : keep record of before
            if DEBUG:
                self._check_sanity()
                decrypter_backup = copy.deepcopy(DECRYPTER)
                self_backup = copy.deepcopy(self)

            # keep a note of quality before change
            old_quadgram_frequency_quality = self._quadgram_frequency_quality

            # apply change now
            DECRYPTER.change(cipher_quadgram[position], quadgram_replacing[position])
            self._change(cipher_quadgram, quadgram_killed, quadgram_replacing, position)

            # for debug purpose : check sanity after changes
            if DEBUG:
                self._check_sanity()

            # evaluate new quadgram quality
            self._set_quadgram_frequency_quality()

            # did the quality improve ?
            if self._quadgram_frequency_quality > old_quadgram_frequency_quality:
                # yes : stop looping : we have improved
                return True

            # this loop cannot be infinite
            attempts_find_improvement_left -= 1
            if attempts_find_improvement_left == 0:
                #  print("Giving up... difficult to improve quadgram quality...")
                return False

            # no improvement so undo
            DECRYPTER.change(cipher_quadgram[position], quadgram_killed[position])
            self._change(cipher_quadgram, quadgram_replacing, quadgram_killed, position)  # pylint:disable=arguments-out-of-order

            # for debug purpose : check we really are back to previous situation
            if DEBUG:
                self._check_sanity()
                assert DECRYPTER == decrypter_backup
                assert self == self_backup
                # check quality is back as it was
                # evaluate new quadgram quality
                self._set_quadgram_frequency_quality()
                assert abs(self._quadgram_frequency_quality - old_quadgram_frequency_quality) < EPSILON_DELTA_FLOAT

            # restore value
            self._quadgram_frequency_quality = old_quadgram_frequency_quality

    def ascend(self) -> None:
        """ ascend : keep climbing until fails to do so """

        while True:

            # keep climbing until fail to do so
            print(".", end='', flush=True)
            succeeded = self._climb()
            if not succeeded:
                print()
                return

    @property
    def quadgram_frequency_quality(self) -> float:
        """ property """
        return self._quadgram_frequency_quality

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Attacker):
            return NotImplemented
        return self._quadgram_allocation_table == other._quadgram_allocation_table and self._reverse_quadgram_allocation_table == other._reverse_quadgram_allocation_table

    def __str__(self) -> str:
        return "\n".join([pprint.pformat(self._quadgram_allocation_table), pprint.pformat(self._reverse_quadgram_allocation_table)])


ATTACKER: typing.Optional[Attacker] = None


def main() -> None:
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', required=False, help='seed random generator to value')
    parser.add_argument('-f', '--frequency', required=True, help='input a file with frequency table for quadgrams (n-letters)')
    parser.add_argument('-d', '--dictionary', required=True, help='dictionary to use')
    parser.add_argument('-l', '--limit', required=False, help='limit in number of words loaded')
    parser.add_argument('-c', '--cipher', required=True, help='cipher to attack')
    parser.add_argument('-D', '--debug', required=False, help='give altitude of cipher taken as clear obtained', action='store_true')
    args = parser.parse_args()

    #  seed = time.time()
    seed = args.seed
    if seed is None:
        seed = time.time()
    random.seed(seed)

    frequency_file = args.frequency
    global NGRAMS_FREQUENCY_TABLE
    NGRAMS_FREQUENCY_TABLE = load_frequency_table(frequency_file)
    #  pprint.pprint(NGRAMS_FREQUENCY_TABLE)

    dictionary_file = args.dictionary
    limit = int(args.limit) if args.limit is not None else None
    global DICTIONARY
    DICTIONARY = Dictionary(dictionary_file, limit)
    #  print(DICTIONARY)
    #  debug(0, DICTIONARY._trie._root)

    debug_mode = args.debug

    cipher_file = args.cipher
    global CIPHER
    CIPHER = Cipher(cipher_file, debug_mode)
    print(f"{CIPHER}")

    global DECRYPTER
    DECRYPTER = Decrypter(CIPHER.cipher_codes)

    global ATTACKER

    if debug_mode:

        ATTACKER = Attacker(debug_mode)
        print("Debug mode !")

        # show quadgram quality
        print(f"{ATTACKER.quadgram_frequency_quality=}")

        # show dictionary quality (and words detected)
        clear = DECRYPTER.apply(CIPHER.cipher_str)
        dictionary_quality, selected_words = DICTIONARY.quality_from_dictionary(clear)
        print(f"{dictionary_quality=}")
        CIPHER.display_result(selected_words)

        return

    best_dictionary_quality_sofar: typing.Optional[float] = None
    while True:

        # start a new session
        ATTACKER = Attacker(debug_mode)
        ATTACKER.ascend()

        # get dictionary quality of result
        clear = DECRYPTER.apply(CIPHER.cipher_str)
        dictionary_quality, selected_words = DICTIONARY.quality_from_dictionary(clear)

        if best_dictionary_quality_sofar is None or dictionary_quality > best_dictionary_quality_sofar:
            print(f"{dictionary_quality=}")
            CIPHER.display_result(selected_words)
            best_dictionary_quality_sofar = dictionary_quality


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
