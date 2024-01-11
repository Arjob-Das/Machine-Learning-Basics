from nltk.corpus import stopwords
import string
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nltk

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
print(stopwords.words('english'))
