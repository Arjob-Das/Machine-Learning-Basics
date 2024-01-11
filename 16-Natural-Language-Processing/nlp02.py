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
print(len(messages))
print(messages[50])
for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                       names=["label", "message"])
print("Head of messages : \n{d}".format(d=messages.head()))

print("Description of messages : \n{d}".format(d=messages.describe()))
print("Group by label : \n{d}".format(d=messages.groupby('label').describe()))

messages['length'] = messages['message'].apply(len)
print("Head of messages with length : \n{d}".format(d=messages.head()))

print("Descriptiom of length of messages : \n{d}".format(
    d=messages['length'].describe()))

print("The longest text message : \n{d}".format(
    d=messages[messages['length'] == 910]['message'].iloc[0]))

mess = 'Sample message! Notice: it has punctuation.'
nopunc = [c for c in mess if c not in string.punctuation]
print(nopunc)
# print(stopwords.words('english'))

nopunc = ''.join(nopunc)
nopunc = nopunc.replace('has', 'has no')
print(nopunc)

print("Nopunc split : \n", nopunc.split())
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

print("Messages Dataframe before processing : \n")

print("Head of messages : \n{d}".format(d=messages.head()))

print("Description of messages : \n{d}".format(d=messages.describe()))
print("Group by label : \n{d}".format(d=messages.groupby('label').describe()))

messages['length'] = messages['message'].apply(len)
print("Head of messages with length : \n{d}".format(d=messages.head()))

print("Descriptiom of length of messages : \n{d}".format(
    d=messages['length'].describe()))

print("Head of messages : \n{d}".format(d=messages.head()))

print("Messages Dataframe after processing : \n")

print("Head of messages : \n{d}".format(
    d=messages['message'].head(5).apply(text_process)))


bow_transformer = CountVectorizer(
    analyzer=text_process).fit(messages['message'])

print(len(bow_transformer.vocabulary_))

mess4 = messages['message'][3]
print(mess4)
bow4 = bow_transformer.transform([mess4])
print(bow4)
print(bow4.shape)
