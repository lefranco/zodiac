#!/usr/bin/python3
# Change the above #! if you want to use python (slower) rather than pypy

import random, re, sys, math
import time
from segment import segment

# Usage: ./crack.py ciphertext
# where ciphertext is (you guessed it) a plaintext file containing the
# ciphertext

# Currently uses a hill-climbing algorithm, which performs surprisingly well.
# TODO In future, simulated annealing is supposed to work a lot better

# Code based upon (basically a fork of)
# http://practicalcryptography.com/media/cryptanalysis/files/ngram_score_1.py

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ASCII_OFFSET = ord('A')
MAX_ITERATIONS = 1000

# Read ciphertext using filename from cli argument
ciphertext = open(sys.argv[1]).read()

# Bigrams list, trigram, etc
class NGrams(object):
    # Load and parse ngrams list
    # File format should be
    # aard 5
    # etc
    def __init__(self, filename):
        self.ngrams = {}

        for line in open(filename):
            char, count = line.split(" ")
            self.ngrams[char] = int(count)

        # The n in ngram
        self.L = len(char)
        # Total number of occurences
        self.N = sum(self.ngrams.values())

        # Turn count into a probability, then log it. Done to avoid rounding
        # errors with small floating point numbers.
        for char in self.ngrams.keys():
            self.ngrams[char] = math.log10(float(self.ngrams[char])/self.N)
        # Don't want prob to be -infty
        self.floor = math.log10(0.01/self.N)

    # Score a given text by comparing it's ngrams
    def score_text(self, text):
        score = 0
        for i in range(len(text)-self.L+1):
            # For each segment of the same length as the ngrams
            # Add it's probability if it's in the list of ngrams
            # Otherwise add the lowest prob
            if text[i:i+self.L] in self.ngrams:
                score += self.ngrams[text[i:i+self.L]]
            else: score += self.floor
        return score

# Given a ciphertext and possible key, decipher the text to return a possible
# plaintext
def decipher(text, key):
    # Find an inverse to the key. e.g. A->B, C->D goes to B->A, D->C
    inverse = [ALPHABET[key.index(i)%26] for i in ALPHABET]
    plaintext = ''
    for char in text:
        # Only want to decode letters
        # ord(char.upper())-ASCII_OFFSET just gives us 0..25 value of letter
        if char.isalpha(): plaintext += inverse[ord(char.upper())-ASCII_OFFSET]
        else: plaintext += char
    return plaintext

# For the moment, just use quadgrams to score as that seems to work really well
# TODO Would adding in lower-n grams, or just using quintgrams give better
# performance (at a guess, quintgrams would, but be much slower)?
quadgrams = NGrams("quadgrams.txt")
def score_key(ciphertext, key):
    # Decipher and remove anything that's not a letter
    plaintext = decipher(ciphertext, key)
    #print(f"{plaintext=}")
    return quadgrams.score_text(re.sub('[^A-Z]', '', plaintext.upper()))

# Ready a plaintext for output
def format_plaintext(plaintext):
    # If there are no spaces in the plaintext, segment it into words using
    # j2kun's library
    # Not 100% accurate at word segmentation, but a lot better than no
    # segmentation at all
    if not (" " in plaintext):
        words = []
        for section in plaintext.upper().split("AND"):
            words.append(" ".join(segment(section)))
        plaintext = " and ".join(words)

    return plaintext.lower()

#random.seed(0)

ND = 0

# Run the whole decryption process multiple times, as it's a non-deterministic
# process
best_key = None
best_score = -99e99
START = time.time()
while 1:
    # Start with an initial key and score it
    key = list(ALPHABET)
    random.shuffle(key)
    score = score_key(ciphertext, key)

    # Only change the key a 1000 times with no score improvement before stopping
    count = 0
    while count < 1000:
        # Create a new key by randomly swapping two positions
        a = random.randint(0,25)
        b = random.randint(0,25)
        new_key = key[:]
        new_key[a], new_key[b] = new_key[b], new_key[a]

        # Score the new key
        new_score = score_key(ciphertext, new_key)
        ND+=1

        #print(f"{count=} {new_score=}")

        # If the new key was better, replace the old one with it
        if new_score > score:
            print("/", end='', flush=True)
            score, key = new_score, new_key[:]


            count = 0 # Restart the counter for detecting no score improvement
        count += 1

    print("-", end='', flush=True)

    # Only use this key if it was better than the last algo iteration
    if score > best_score:

        best_score = score
        best_key = key
        print()
        print (format_plaintext(decipher(ciphertext, best_key)))
        now = time.time()
        speed = ND/(now - START)
        print(f"{speed=}")
        key_="".join(key)
        print(f"{key_=} {score=}")

    print()