#!/usr/bin/python3

#pylint: disable=global-statement
#pylint: disable=too-many-statements,too-few-public-methods,too-many-branches,too-many-locals
#pylint: disable=too-many-nested-blocks,too-many-return-statements

"""
Decodes an 'homophonic cipher.
This is a cipher where :
  - one letter from plain text can be coded in one or more codes (usually to have even frequency
    of codes in cipher)
  - the seperation between words is not shown
The method implemented here uses a dictionnary, I.e. a list of words.
It can only succeed if all the words from plain text are in the dictionnary.
"""


import sys
import time
import copy
import argparse
import collections
import enum
import random

import cProfile
#import pstats

DEBUG = True
#DEBUG = False
PROGRESS = False

# constant
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# to adjust ?
MOST_FREQUENT_LETTERS = 'ETAOINSHRDLU'


# global
DICTIONNARY = None
CIPHER = None

class Possibility(collections.namedtuple('Possibility', ['position', 'word'])):
    """
        position : where in the grid it can be
        word : the word
    """
    __slots__ = ()
    def __repr__(self):
        """ for debug """
        return "poss : position={} word=/{}/".format(self.position, self.word)

class WordList(object):
    """ Word list object (singleton) a dictionnary (python) of pattern -> list of words"""

    def __init__(self, filename):

        before = time.time()

        self._words = set()
        self.orderword = dict()

        self.longest_word = 0

        if filename:
            self.register_file(filename)
        after = time.time()
        print("Word list file '{}' loaded in {} seconds".format(filename, after - before))

    def register_word(self, word, line_num):
        """ Enter a word from file in our dictionnary """

        assert not [l for l in word if l not in ALPHABET], \
        'ERROR : bad word found in dictionnary line ' + str(line_num) + ' ' + '<' + word + '>'
        if word in self._words:
            print("Warning : duplicated word '{}' for dictionnary line {}".format(word, line_num))
            return

        # added in set of words
        self._words.add(word)

        # order (the least, the most frequent to consider first)
        self.orderword[word] = len(self._words)

        # need to know longest word
        if len(word) > self.longest_word:
            self.longest_word = len(word)

    def register_file(self, filename):
        """ Enter all the words from a file """

        line_num = 1
        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                if line:
                    word = line
                    word = word.upper()
                    self.register_word(word, line_num)
                line_num += 1

    def contains_word(self, word):
        """ Just for checking word list has no duplicates """
        return word in self._words

    def __str__(self):
        """ for debug """
        return " ".join([word for word in self._words])

class Allocation(object):
    """ An simple object giving association : cipher code -> plain letter """

    def __init__(self, word_cipher, word_plain):
        self.table = dict()
        for(code_cipher, letter_plain) in zip(word_cipher, word_plain):
            if code_cipher in self.table:
                assert self.table[code_cipher] == letter_plain
                continue
            self.table[code_cipher] = letter_plain

    def compatible(self, other):
        """ True if are compatible = do not conflict """

        # same cipher must give same plain
        inter = self.table.keys() & other.table.keys()
        if [self.table[l] for l in inter] != [other.table[l] for l in inter]:
            return False

        return True

    def merge(self, other):
        """ Makes a bigger allocation with both """

        self.table.update(other.table)

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.table = copy.copy(self.table)
        return other

    def __str__(self):
        sprint = ""
        for code in sorted(self.table):
            sprint += "{}~{} ".format(code, self.table[code])
        return sprint

class Cipher(object):
    """ Cipher object (singleton) is the cipher message """

    def __init__(self, filename):

        assert DICTIONNARY, "ERROR : Define word list before cipher"

        # solutions found
        self.solutions = set()

        # codes
        self.codes = []

        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                if line:

                    for code_str in line.split(' '):
                        code = int(code_str)
                        self.register(code)


    def register(self, code):
        """ Enter this word in the message """

        self.codes.append(code)

    def display_solution(self, allocation, partial=False):
        """ prints a solution that was found """

        if allocation in self.solutions:
            # do not repeat
            return

        self.solutions.add(allocation)

        if not partial:
            print("")
            print("Solution found : ", end='')

        for code in self.codes:
            plain_letter = allocation.table[code].lower()
            print("{}".format(plain_letter), end='')
        print("")

    def __str__(self):
        """ for debug """
        return " ".join([str(code) for code in self.codes])

class Element(object):
    """ ... """

    def __init__(self, code):
        self.code = code
        self.letter = None
        self.word = None

    def __str__(self):
        """ for debug """
        return self.letter if self.letter else self.code

class Grid(object):
    """ ... """

    def __init__(self, codes):
        self.table = []
        for code in codes:
            element = Element(code)
            self.table.append(element)

    def solved(self):
        """ ... """
        return all([elem.word for elem in self.table])

    def failed(self):
        """ ... """
        return all([elem.letter is None for elem in self.table]) and \
               any([elem.word is None for elem in self.table])

    def apply_allocation(self, allocation):
        """ Puts the word there and apply the consequences """

        # other occurences of codes
        for element in self.table:
            element.letter = allocation.table[element.code]
            element.word = None

    def apply_word(self, position, word):
        """ Puts the word there and apply the consequences """

        # two same words get a different copy of word
        word_copy = copy.copy(word)

        # apply for word
        for offset, letter in enumerate(word):

            # put letter there
            element = self.table[position + offset]
            if element.letter:
                assert element.letter == letter
            else:
                element.letter = letter

            # element in word
            assert not element.word
            element.word = word_copy

    def possible_word(self, position, word):
        """ Returns True if the word can go there """

        # apply for word
        for offset, letter in enumerate(word):

            # if letter there must be same
            element = self.table[position + offset]

            # there must ne already be a word there
            if element.word:
                return False

            if element.letter and element.letter != letter:
                return False

        return True

    def words_found(self):
        """ ... """

        result = []
        for start_pos in range(len(self.table)-1):
            for length in range(1, min(DICTIONNARY.longest_word, len(self.table) - start_pos + 1)):

                # test all letters and not in words
                extract_letters = [elem.letter for elem in self.table[start_pos: start_pos+length]]
                assert all(extract_letters)

                word = ''.join(extract_letters).upper()
                if DICTIONNARY.contains_word(word):
                    possibility = Possibility(position=start_pos, word=word)
                    result.append(possibility)

        return sorted(result, key=lambda p: len(p.word), reverse=True)

    def __str__(self):
        """ for debug """

        class ElemType(enum.Enum):
            """ ... """
            init = 1
            code = 2
            letter = 3
            word = 4

        sprint = ''
        pos = 0
        last = ElemType.init
        while pos < len(self.table):
            element = self.table[pos]
            if element.word:
                sprint += "[{}]".format(element.word)
                pos += len(element.word)
                last = ElemType.word
            elif element.letter:
                if last in [ElemType.code]:
                    sprint += "/"
                sprint += element.letter
                pos += 1
                last = ElemType.letter
            else:
                if last in [ElemType.letter, ElemType.code]:
                    sprint += "/"
                sprint += "{}".format(element.code)
                pos += 1
                last = ElemType.code
        return sprint

class Phenotype(object):
    """ ... """

    def __init__(self):
        codes = []
        letters = []
        for code in sorted(set(CIPHER.codes)):
            letter = random.choice(ALPHABET)
            codes.append(code)
            letters.append(letter)
        self.allocation = Allocation(codes, letters)

        self.grid = Grid(CIPHER.codes)
        self.grid.apply_allocation(self.allocation)

        self.words = self.grid.words_found()

        self.word_score = 0
        for poss in self.words:
            #print(poss)
            # test required because two words may overlap
            # todo : sort the words to consider the longest ones first
            if self.grid.possible_word(poss.position, poss.word):
                self.grid.apply_word(poss.position, poss.word)
                len_word = len(poss.word)
                self.word_score += len_word*len_word

    def mating(self, other):
        """ ..."""
        # TBD self + other --> sibling
        pass

    def __str__(self):
        sprint = ""
        sprint += "allocation : "
        sprint += str(self.allocation)
        sprint += "\n"
        sprint += "grid : "
        sprint += str(self.grid)
        sprint += "\n"
        sprint += "score : "
        sprint += str(self.word_score)
        sprint += "\n"
        return sprint

def solve():
    """ Solver. """

    score_table = dict()
    for i in range(2000):
        if i%1000==0:
            print(".",end='', flush=True)
        phenotype = Phenotype()
        score = phenotype.word_score
        if score not in score_table:
            score_table[score] = []
        score_table[score].append(phenotype)

    for score, phenos in sorted(score_table.items(), reverse=True)[0:10]:
        for pheno in phenos:
            print("########")
            print("{} : {}".format(score, pheno))


def main():
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='force seed value (def = from time system call)')
    parser.add_argument('-c', '--cipher', required=True, help='cipher to attack')
    parser.add_argument('-d', '--dictionnary', required=True,
                        help='dictionnary to use')
    args = parser.parse_args()
    #print(args)

    if args.seed:
        seed = int(args.seed)
    else:
        seed = time.time()
    random.seed(seed)

    dictionnary_file = args.dictionnary
    global DICTIONNARY
    DICTIONNARY = WordList(dictionnary_file)

    if DEBUG:
        print("DICTIONNARY:", file=sys.stderr)
        print(DICTIONNARY, file=sys.stderr)

    cipher_file = args.cipher
    global CIPHER
    CIPHER = Cipher(cipher_file)

    if DEBUG:
        print("CIPHER:", file=sys.stderr)
        print(CIPHER, file=sys.stderr)

    solve()

if __name__ == '__main__':

    # this to know how long it takes
    START = time.time()

    # this if script too slow and profile it
    PR = cProfile.Profile()

    #PR.enable()
    main()
    #PR.disable()

    # stats
    #PS = pstats.Stats(PR)
    #PS.strip_dirs()
    #PS.sort_stats('time')
    #PS.print_stats() # uncomment to have profile stats

    # how long it took
    DONE = time.time()
    print("Time elapsed : {:f}".format(DONE - START))

    sys.exit(0)
