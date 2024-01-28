from sklearn.feature_extraction.text import CountVectorizer
import time
from nltk.corpus import stopwords
import string
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import os

plt.ion()

# nltk.download()
messages = [line.rstrip() for line in open(
    'smsspamcollection/SMSSpamCollection')]

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                       names=["label", "message"])
print("Head of messages : \n{d}".format(d=messages.head()))

messages['length'] = messages['message'].apply(len)

print("Head of messages with length : \n{d}".format(d=messages.head()))

mess = 'Sample message! Notice: it has punctuation.'
nopunc = [c for c in mess if c not in string.punctuation]

nopunc = ''.join(nopunc)
nopunc = nopunc.replace('has', 'has no')
print(nopunc)

clean_mess = [word for word in nopunc.split() if word.lower()
              not in stopwords.words('english')]
print("After removing stopwords : \n", clean_mess)


def text_process(mess):
    """ 
    1. remove punc
    2. remove stop words
    3. return list of clean text words
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clear_mess = [word for word in nopunc.split() if word.lower()
                  not in stopwords.words('english')]
    return clear_mess


time.sleep(2)

os.system('cls')

messages['length'] = messages['message'].apply(len)

bow_transformer = CountVectorizer(
    analyzer=text_process).fit(messages['message'])

print(len(bow_transformer.vocabulary_))

mess4 = messages['message'][3]
print(mess4)
bow4 = bow_transformer.transform([mess4])
print(bow4)
print(bow4.shape)
