#!/usr/bin/python3

#pylint: disable=global-statement
#pylint: disable=too-many-statements,too-few-public-methods,too-many-branches,too-many-locals

"""
Decodes an 'aristocrat'.
This is a cipher where :
  -  there is a strict correspondance one letter cipher / one letter plain
  - the seperation between words is shown
The method implemented here uses a dictionnary, I.e. a list of words.
It can only succeed if all the words from plain text are in the dictionnary.
"""

import sys
import time
import copy
import argparse

import cProfile
import pstats

DEBUG = False
PROGRESS = False

# constant
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# global
DICTIONNARY = None
CIPHER = None
SEARCH_TABLE = None
SHOW_ALL = False

class Pattern(object):
    """ The pattern of a word that reflects the structure, the duplicated letters """

    def __init__(self, word):

        # length of word
        self.length = len(word)

        # nb letters of word
        self.nbletter = len(set(l for l in word))

        # shape itself
        occ = dict()
        for (pos, let) in enumerate(word):
            if not let in occ:
                occ[let] = []
            occ[let].append(pos)

        self.shape = tuple(sorted(occ.values()))

    def __hash__(self):
        return hash(str(self.length)+ "/" + str(self.nbletter) + "/" + str(self.shape))

    def __eq__(self, other):

        if self.length != other.length:
            return False

        if self.nbletter != other.nbletter:
            return False

        assert len(self.shape) == len(other.shape), 'ERROR : internal error'

        for (elt1, elt2) in zip(self.shape, other.shape):
            if elt1 != elt2:
                return False
        return True

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return self != other

    def __str__(self):
        """ for debug """
        pattdesc = dict()
        for elt in self.shape:
            for pos in elt:
                pattdesc[pos] = self.shape.index(elt)
        return "len={} nblet={} patt=/{}/".format(self.length, self.nbletter,
                                                  " ".join(["_{}".format(n)
                                                            for n in pattdesc.values()]))

class WordList(object):
    """ Word list object (singleton) a dictionnary (python) of pattern -> list of words"""

    def __init__(self, filename):

        before = time.time()

        self._entries = dict()
        self._words = set()
        self.orderword = dict()

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

        # added in pattern driven table
        patt = Pattern(word)
        if patt not in self._entries:
            self._entries[patt] = []
        self._entries[patt].append(word)

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

    def matching_words(self, word):
        """ Gives the words from our word list that have same pattern as the word """

        patt = Pattern(word)
        if patt not in self._entries:
            return None
        return self._entries[patt]

    def contains_word(self, word):
        """ Just for checking word list has no duplicates """
        return word in self._words

    def __str__(self):
        """ for debug """
        ret = ""
        for patt in sorted(self._entries, key=lambda p: (p.length, p.nbletter)):
            ret += "{} : {}\n".format(patt, sorted(self._entries[patt]))
        return ret

class Allocation(object):
    """ An simple object giving association : cipher letter -> plain letter """

    def __init__(self, word_cipher, word_plain):
        self.table = dict()
        for(letter_cipher, letter_plain) in zip(word_cipher, word_plain):
            if letter_cipher in self.table:
                assert self.table[letter_cipher] == letter_plain
                continue
            self.table[letter_cipher] = letter_plain

    def compatible(self, other):
        """ True if are compatible = do not conflict """

        # same cipher must give same plain
        inter = self.table.keys() & other.table.keys()
        if [self.table[l] for l in inter] != [other.table[l] for l in inter]:
            return False

        # different ciphers must not give same plain
        if set([self.table[l] for l in self.table.keys() - inter]) & \
           set([other.table[l] for l in other.table.keys() - inter]):
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
        for letter in sorted(self.table):
            sprint += "{}~{} ".format(letter, self.table[letter])
        return sprint

class Cipher(object):
    """ Cipher object (singleton) is the cipher message """

    def __init__(self, filename):

        assert DICTIONNARY, "ERROR : Define word list before cipher"

        # the list of words of the cipher in the order they appear
        self.sequence = []

        # same without the excluded ones
        self.sequence_considered = []

        # the list of words of the cipher but no duplicated
        self.unique_sequence = []

        # solutions found
        self.solutions = set()

        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                if line:

                    for word in line.split(' '):
                        word = word.upper()
                        self.register(word)


        # init the sequence of words to consider
        self.sequence_considered = copy.copy(self.sequence)

        # list if words in cipher in order of attack
        self.determine_unique()

    def register(self, word):
        """ Enter this word in the message """

        assert not [l for l in word if l not in ALPHABET],\
        'ERROR : impossible word found in cipher : ' + word

        # inserted in sequence (for display)
        self.sequence.append(word)

    def ignore(self, word):
        """ Remove this word from the message """

        print("Ignoring word '{}' from cipher...".format(word))
        if word not in self.sequence_considered:
            print("ERROR : no such word as '{}' in cipher".format(word))
            exit(1)
        while word in self.sequence_considered:
            self.sequence_considered.remove(word)

        self.determine_unique()

    def determine_unique(self):
        """ Updates the unique list of cipher words """
        self.unique_sequence = sorted(list(set(self.sequence_considered)), \
                                          key=len, \
                                          reverse=True)

    def check_foundable(self):
        """ check all non ignored cipher words may match in dictionary """

        for word_cipher in self.unique_sequence:
            # check this word has a pattern in DICTIONNARY
            assert DICTIONNARY.matching_words(word_cipher),\
            'ERROR : This item does not match any word from the dictionnary : ' + word_cipher

    def display_solution(self, allocation, partial=False):
        """ prints a solution that was found """

        if allocation in self.solutions:
            # do not repeat
            return

        self.solutions.add(allocation)

        if not partial:
            print("")
            print("Solution found : ", end='')

        for word_cipher in self.sequence:
            if word_cipher not in self.sequence_considered:
                print('[', end='')
            for letter_cipher in word_cipher:
                if letter_cipher in allocation.table:
                    plain_letter = allocation.table[letter_cipher].lower()
                else:
                    plain_letter = '?'
                print("{}".format(plain_letter), end='')
            if word_cipher not in self.sequence_considered:
                print(']', end='')
            print(" ", end='')
        print("")

    def __str__(self):
        """ for debug """
        return "RAW SEQ:\n\t" + " ".join(self.sequence) + "\n" + \
               "UNIQ SEQ:\n\t" +  " ".join(self.unique_sequence) + "\n"

class SearchTable(object):
    """ Class with all static information to speed up the decoding """

    def __init__(self):

        assert DICTIONNARY, "ERROR : Define word list before search table"
        assert CIPHER, "ERROR : Define cipher before search table"

        self.possible = dict()
        for word_cipher in CIPHER.unique_sequence:
            self.possible[word_cipher] = dict()
            for word_plain in DICTIONNARY.matching_words(word_cipher):
                allocation = Allocation(word_cipher, word_plain)
                self.possible[word_cipher][word_plain] = allocation

    def __str__(self):
        """ for debug """
        sprint = ""
        for word_cipher in sorted(self.possible):
            sprint += word_cipher
            sprint += ":\n"
            for word_plain in self.possible[word_cipher]:
                sprint += "\t"
                sprint += word_plain
                sprint += " : "
                sprint += str(self.possible[word_cipher][word_plain])
                sprint += "\n"
            sprint += "\n"
        return sprint

def solve():
    """ Solver. """

    def solve_rec(cipherwordlist, already_allocations, possible_words, solutions_sofar):
        """ Recurse, as usual... """

        # are we done ?
        if not cipherwordlist:
            if not already_allocations in solutions_sofar:
                solutions_sofar.add(already_allocations)
                CIPHER.display_solution(already_allocations)
            return True

        # select the cipher word we attack - this is the clever part 1 ;-)
        wordciphers_sorted = sorted(possible_words.keys(), key=lambda w: len(possible_words[w]))
        wordcipher_considered = wordciphers_sorted[0]

        depth = len(CIPHER.unique_sequence) - len(cipherwordlist)
        nonlocal depth_max
        if depth > depth_max:
            depth_max = depth
            print("{}/{}({}) ".format(depth, len(CIPHER.unique_sequence), wordcipher_considered), \
                  end='', flush=True)
            # for the fun : to follow progress on screen
            if PROGRESS:
                print("")
                print("---")
                print(" ".join(CIPHER.sequence))
                CIPHER.display_solution(already_allocations, partial=True)
                print("---")

        # consider words in order of word list : first are more frequent so are considered first
        word_plain_possible = sorted(possible_words[wordcipher_considered],
                                     key=lambda w: DICTIONNARY.orderword[w])
        for word_plain in word_plain_possible:

            if DEBUG:
                print(" " * depth, end='', file=sys.stderr)
                print(wordcipher_considered, end='', file=sys.stderr)
                print(" = ", end='', file=sys.stderr)
                print(word_plain.lower(), file=sys.stderr)

            # try to go deeper
            alloc = SEARCH_TABLE.possible[wordcipher_considered][word_plain]
            if not already_allocations.compatible(alloc):
                continue

            # calculate new cipherwordlist
            cipherwordlist2 = copy.copy(cipherwordlist)
            cipherwordlist2.remove(wordcipher_considered)

            # calculate new already_allocations
            already_allocations2 = copy.copy(already_allocations)
            already_allocations2.merge(alloc)

            # calculate new possible_words
            possible_words2 = copy.deepcopy(possible_words)
            del possible_words2[wordcipher_considered]

            # prune the possibilities - this is the clever part 2 ;-)
            for word_cipher in possible_words2.keys():
                remove_list = []
                for word_plain2 in possible_words2[word_cipher]:
                    alloc2 = SEARCH_TABLE.possible[word_cipher][word_plain2]
                    if not already_allocations2.compatible(alloc2):
                        remove_list.append(word_plain2)
                for word_plain2 in remove_list:
                    possible_words2[word_cipher].remove(word_plain2)

            if solve_rec(cipherwordlist2, already_allocations2, possible_words2, solutions_sofar):
                if not SHOW_ALL:
                    return True

        return False

    depth_max = 0

    void_alloc = Allocation([], [])
    possible_words = dict()
    for word_cipher in CIPHER.unique_sequence:
        possible_words[word_cipher] = set(SEARCH_TABLE.possible[word_cipher].keys())

    solutions_sofar = set()
    solve_rec(CIPHER.unique_sequence, void_alloc, possible_words, solutions_sofar)
    if not solutions_sofar:
        print("Failed to solve !")
        exit(1)

def main():
    """ main """


    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cipher', required=True, help='cipher to attack')
    parser.add_argument('-d', '--dictionnary', required=True,
                        help='dictionnary to use')
    parser.add_argument('-i', '--ignore', nargs='+',
                        help='cipher words to ignore because probably not in dictionnary')
    parser.add_argument('-a', '--all', action='store_true', help='tries out all the solutions')
    args = parser.parse_args()
    #print(args)

    dictionnary_file = args.dictionnary
    global DICTIONNARY
    DICTIONNARY = WordList(dictionnary_file)

    if DEBUG:
        print("DICTIONNARY:", file=sys.stderr)
        print(DICTIONNARY, file=sys.stderr)

    cipher_file = args.cipher
    global CIPHER
    CIPHER = Cipher(cipher_file)

    global SHOW_ALL
    if args.all:
        SHOW_ALL = True

    if args.ignore:
        for ignored_cipherword in args.ignore:
            CIPHER.ignore(ignored_cipherword.upper())

    # check no impossible words
    CIPHER.check_foundable()

    if DEBUG:
        print("CIPHER:", file=sys.stderr)
        print(CIPHER, file=sys.stderr)

    # search table
    global SEARCH_TABLE
    SEARCH_TABLE = SearchTable()

    if DEBUG:
        print("SEARCH_TABLE:", file=sys.stderr)
        print(SEARCH_TABLE, file=sys.stderr)

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
