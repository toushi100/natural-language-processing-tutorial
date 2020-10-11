import pandas as pd
from textblob import TextBlob
import numpy as np
import math
import matplotlib.pyplot as plt
data = pd.read_pickle('corpus')
print(data)
pol = lambda x : TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = data['transcript'].apply(pol)
data['subjectivity'] = data['transcript'].apply(sub)
print(data)

plt.rcParams['figure.figsize'] = [10,8]

for index , comedian in enumerate(data.index):
    x = data.polarity.loc[comedian]
    y = data.subjectivity.loc[comedian]
    plt.scatter(x,y,color='blue')
    plt.text(x+.001,y+.001,data['full_name'][index], fontsize=10)
    plt.xlim(-.01,.12)

plt.title('Sentiment Analysis',fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)
plt.show()

def split_text(text, n=10):
    length = len(text)
    size = math.floor(length/n)
    start = np.arange(0,length,size)
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list

print(data)
list_piece = []
for t in data.transcript:
    split = split_text(t)
    list_piece.append(split)

print(len(list_piece))
print(len(list_piece[0]))

polarity_transcript = []
for lp in list_piece:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)

print(polarity_transcript)
plt.plot(polarity_transcript[0])
plt.title(data['full_name'].index[0])
plt.show()

plt.rcParams['figure.figsize'] = [16, 12]

for index, comedian in enumerate(data.index):
    plt.subplot(3, 4, index + 1)
    plt.plot(polarity_transcript[index])
    plt.plot(np.arange(0, 10), np.zeros(10))
    plt.title(data['full_name'][index])
    plt.ylim(ymin=-.2, ymax=.3)

plt.show()