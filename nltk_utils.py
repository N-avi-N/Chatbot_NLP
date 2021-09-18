import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# initialize the stemmer
stemmer = PorterStemmer()


# tokenize sentences
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# stem a word
def stem(word):
    return stemmer.stem(word.lower())

# convert sentence to bag of words
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # create a empty bag of zeroes same as size of vocab
    # update the index of words in sentence as 1 in the bag
    bag = np.zeros(len(all_words), dtype=np.float32)

    # update the present words index to '1'
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

# Testing Functions

# a = 'Would you like a cookie ?'
# print(a)
# a = tokenize(a)
# print(a)
#
# words = ['Created', 'creating', 'creates']
# stemmed_words = [stem(word) for word in words]
# print(stemmed_words)

sentence = ['hello', 'how', 'are', 'you']
words = ['hello', 'hi', 'i', 'you', 'bye', 'cool', 'thank']
bag = bag_of_words(sentence, words)
print(bag)