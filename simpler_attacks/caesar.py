#!/usr/bin/python3

#pylint: disable=global-statement
#pylint: disable=too-many-statements,too-few-public-methods,too-many-branches,too-many-locals

"""
Decodes a 'Casear' cipher.
Letters are all shifted of the same value (between 1 and 25).
Just displays 25 possibilities
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
CIPHER = None

class Cipher(object):
    """ Cipher object (singleton) is the cipher message """

    def __init__(self, filename):

        # the list of words of the cipher in the order they appear
        self.plain_sequence = []

        with open(filename) as filepointer:
            for line in filepointer:
                line = line.rstrip('\n')
                if line:

                    for word in line.split(' '):
                        word = word.upper()
                        self.register(word)

    def register(self, word):
        """ Enter this word in the message """

        assert not [l for l in word if l not in ALPHABET],\
        'ERROR : impossible word found in cipher : ' + word

        # inserted in sequence (for display)
        self.plain_sequence.append(word)


    def attempt(self, allocation, partial=False):
        """ prints a solution that was found """

        for word_cipher in self.plain_sequence:
            for letter_cipher in word_cipher:
                if letter_cipher in allocation.table:
                    plain_letter = allocation.table[letter_cipher].lower()
                else:
                    plain_letter = '?'
                print("{}".format(plain_letter), end='')
            print(" ", end='')
        print("")

    def __str__(self):
        """ for debug """
        return "PLAIN SEQ:\n\t" + " ".join(self.plain_sequence) + "\n" + \
               "UNIQ SEQ:\n\t" +  " ".join(self.unique_sequence) + "\n"

class Allocation(object):
    """ An simple object saying cipher letter -> plain letter """

    def __init__(self, shift):
        self.table = dict()
        for n, letter_cipher in enumerate(ALPHABET):
            p = (n + shift) % len(ALPHABET)
            letter_plain = ALPHABET[p]
            self.table[letter_cipher] = letter_plain

    def __str__(self):
        sprint = ""
        for letter in sorted(self.table):
            sprint += "{}~{} ".format(letter, self.table[letter])
        return sprint

def solve():
    """ Solver. """

    for shift in range(1,len(ALPHABET)):
        allocation = Allocation(shift)
        CIPHER.attempt(allocation)

def main():
    """ main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cipher', required=True, help='cipher to attack')
    args = parser.parse_args()
    #print(args)

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
