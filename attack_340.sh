sudo nice -n 5 pypy3 ./homophonic.py -p 10 -i ./data/english/raw/ioc.txt -n ./data/english/freq/english_quintgrams.txt -d ./data/english/dict/english_words.txt -l ./data/english/freq/english_monograms.txt -c ./ciphers/homophones/zodiac/cipher-340-transposed.txt  -o result.txt