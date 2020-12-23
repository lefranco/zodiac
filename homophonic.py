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
DEBUG = False

EPSILON = 0.00001

COUNTER_MAX = 5000
COUNTER_MAX2 = 1000

ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]

NGRAMS = 4


def load_frequency_table(filename: str) -> typing.Dict[str, int]:
    """ load_frequency_table """

    before = time.time()

    frequency_table: typing.Dict[str, int] = dict()
    with open(filename) as filepointer:
        for line in filepointer:
            line = line.rstrip('\n')
            quadgram_read, frequency_str = line.split()
            quadgram = quadgram_read.lower()
            frequency = int(frequency_str)
            frequency_table[quadgram] = frequency

    coverage = (len(frequency_table) / (len(ALPHABET) ** NGRAMS)) * 100
    print(f"Frequency tables covers {coverage:.2f}% of possibilities")

    # covering the rest
    default_frequency = min(frequency_table.values())
    for letters in itertools.product(ALPHABET, repeat=NGRAMS):
        quadgram = ''.join(letters)
        if quadgram not in frequency_table:
            frequency_table[quadgram] = default_frequency

    after = time.time()
    elapsed = after - before
    print(f"N-Gram frequency file '{filename}' loaded in {elapsed:2.2f} seconds")

    return frequency_table


NGRAMS_FREQUENCY_TABLE: typing.Dict[str, int] = dict()


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

        def heuristic(word : str) -> float:
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
        ret = ""
        for word, log_freq in self._log_frequency_table.items():
            ret += f"{word} {log_freq} l / log={len(word)/log_freq}\n"
        return ret


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
        self._quadgram_interaction_table: typing.Dict[str, typing.Dict[int, typing.Set[str]]] = collections.defaultdict(dict)
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
    def quadgram_interaction_table(self) -> typing.Dict[str, typing.Set[str]]:
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

        # make initial random allocation
        assert CIPHER is not None
        if debug_mode:
            allocation = {k: k for k in CIPHER.cipher_codes}
        else:
            allocation = {k: random.choice(ALPHABET) for k in CIPHER.cipher_codes}

        # put it in crypter
        assert DECRYPTER is not None
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
        assert DICTIONARY is not None
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

        assert DEBUG

        # check tables coherent
        for cipher_quad, clear_quad in self._quadgram_allocation_table.items():
            assert cipher_quad in self._reverse_quadgram_allocation_table[clear_quad]

        for clear_quad, cipher_quads in self._reverse_quadgram_allocation_table.items():
            for cipher_quad in cipher_quads:
                assert self._quadgram_allocation_table[cipher_quad] == clear_quad

        # check tables coherent with table in cipher
        assert DECRYPTER is not None
        for cipher_quad, clear_quad in self._quadgram_allocation_table.items():
            assert DECRYPTER.apply(cipher_quad) == clear_quad

    def _change(self, cipher_quadgram: str, quadgram_killed: str, quadgram_replacing: str) -> None:
        """ change : we replace for 'cipher_quadgram' corresponding 'quadgram_killed' by 'quadgram_replacing' """

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
        assert CIPHER is not None
        assert DECRYPTER is not None
        for impacted_cipher_quadgram in CIPHER.quadgram_interaction_table[cipher_quadgram]:
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
        occurences_clear = dict()
        for clear_quadgram, ciphers_quadgrams in self._reverse_quadgram_allocation_table.items():
            occurences_clear[clear_quadgram] = sum([len(CIPHER.quadgram_occurence_table[c]) for c in ciphers_quadgrams])

        total_occurences_clear = sum(occurences_clear.values())
        frequency_clear = {q: occurences_clear[q] / total_occurences_clear for q in occurences_clear}

        occurences_dict = {q: NGRAMS_FREQUENCY_TABLE[q] for q in occurences_clear}
        total_occurences_dict = sum(occurences_dict.values())
        frequency_dictionary = {q: occurences_dict[q] / total_occurences_dict for q in occurences_dict}

        self._quadgram_frequency_quality = - sum([abs(frequency_clear[q] - frequency_dictionary[q]) for q in occurences_clear])

    def climb(self, words_show: bool) -> None:
        """ climb : try to improve things... """

        if DEBUG:
            self._check_sanity()

        # select candidate to change

        # build table to choose from : clear quadgram table
        # a quadgram is all the more likely to be chosen that it is less frequent
        sum_frequencies = sum([NGRAMS_FREQUENCY_TABLE[q] for q in self._quadgram_allocation_table.values()])
        temp_rating = {q: sum_frequencies - NGRAMS_FREQUENCY_TABLE[q] for q in self._quadgram_allocation_table.values()}

        counter = 0
        while True:

            # randomly select a 'quadgram_killed' in the table
            quadgram_killed_list: typing.List[str] = random.choices(list(temp_rating.keys()), weights=list(temp_rating.values()))
            quadgram_killed = quadgram_killed_list[0]

            counter2 = 0
            while True:

                # randomly select a 'cipher_quadgram' corresponding (usually there is only one)
                cipher_quadgram: str = random.choice(list(self._reverse_quadgram_allocation_table[quadgram_killed]))

                # randomly select a 'quadgram_replacing' : what do we change this quadgram to ?
                # a quadgram is all the more likely to be chosen that it is more frequent
                quadgram_replacing_list: typing.List[str] = random.choices(list(NGRAMS_FREQUENCY_TABLE.keys()), weights=list(NGRAMS_FREQUENCY_TABLE.values()))
                quadgram_replacing = quadgram_replacing_list[0]

                # must respect pattern
                if self._accept(cipher_quadgram, quadgram_replacing):
                    break

                counter2 += 1
                if counter2 > COUNTER_MAX2:
                    print()
                    print("Giving up... difficult to get a replacing quadgram...")
                    assert False
                    break

            if DEBUG:
                self._check_sanity()

            # apply change in decrypter
            assert DECRYPTER is not None

            # for debug purpose
            if DEBUG:
                decrypter_backup = copy.deepcopy(DECRYPTER)

            DECRYPTER.change(cipher_quadgram, quadgram_replacing)

            # keep a note of quality before change
            old_quadgram_frequency_quality = self._quadgram_frequency_quality

            # for debug purpose
            if DEBUG:
                self_backup = copy.deepcopy(self)

            # change now
            self._change(cipher_quadgram, quadgram_killed, quadgram_replacing)

            if DEBUG:
                self._check_sanity()

            # evaluat new quadgram quality
            self._set_quadgram_frequency_quality()

            # did the quality improve ?
            if self._quadgram_frequency_quality > old_quadgram_frequency_quality:
                # yes : stop looping
                break

            # this loop cannot be infinite
            counter += 1
            if counter > COUNTER_MAX:
                print()
                print("Giving up... difficult to improve quadgram quality...")
                assert False
                break

            # no improvement so undo
            DECRYPTER.change(cipher_quadgram, quadgram_killed)

            self._change(cipher_quadgram, quadgram_replacing, quadgram_killed)  # pylint:disable=arguments-out-of-order

            if DEBUG:
                self._check_sanity()

            # for debug purpose : check we really are back to previous situation
            if DEBUG:
                assert DECRYPTER == decrypter_backup
                assert self == self_backup

            # for debug purpose
            if DEBUG:
                # check quality is back as it was
                # evaluat new quadgram quality
                self._set_quadgram_frequency_quality()
                assert abs(self._quadgram_frequency_quality - old_quadgram_frequency_quality) < EPSILON

            # restore value
            self._quadgram_frequency_quality = old_quadgram_frequency_quality

        print(f"quadgram quality={self._quadgram_frequency_quality}")

        # get dictionary quality of new situation
        assert CIPHER is not None
        clear = DECRYPTER.apply(CIPHER.cipher_str)

        if words_show:

            assert DICTIONARY is not None
            new_dict_quality, selected_words = DICTIONARY.quality_from_dictionary(clear)

            # change quality
            self._dictionary_quality = new_dict_quality

            # best visual experience
            CIPHER.display_result(selected_words)

        else:

            print(f"{clear}")

        print()

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
    parser.add_argument('-f', '--frequency', required=True, help='input a file with frequency table for quadgrams (n-letters)')
    parser.add_argument('-d', '--dictionary', required=True, help='dictionary to use')
    parser.add_argument('-l', '--limit', required=False, help='limit in number of words loaded')
    parser.add_argument('-c', '--cipher', required=True, help='cipher to attack')
    parser.add_argument('-w', '--words_show', required=False, help='show words of clear in real time', action='store_true')
    parser.add_argument('-D', '--debug', required=False, help='give altitude of cipher taken as clear obtained', action='store_true')
    args = parser.parse_args()

    seed = time.time()
    #  seed = 0
    random.seed(seed)

    frequency_file = args.frequency
    global NGRAMS_FREQUENCY_TABLE
    NGRAMS_FREQUENCY_TABLE = load_frequency_table(frequency_file)

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

    global DECRYPTER
    DECRYPTER = Decrypter(CIPHER.cipher_codes)

    global ATTACKER
    ATTACKER = Attacker(debug_mode)

    if debug_mode:

        print("Debug mode !")

        # show quadgram quality
        print(f"{ATTACKER._quadgram_frequency_quality=}")

        # show dictionary quality (and words detected)
        clear = DECRYPTER.apply(CIPHER.cipher_str)
        dictionary_quality, selected_words = DICTIONARY.quality_from_dictionary(clear)
        print(f"{dictionary_quality=}")
        CIPHER.display_result(selected_words)

        print()
        return

    words_show = args.words_show

    while True:
        ATTACKER.climb(words_show)


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
