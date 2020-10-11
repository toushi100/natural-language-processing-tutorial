import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import pickle

data = pd.read_pickle('dtm.pkl')
data = data.transpose()

top_dict ={}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c] = list(zip(top.index,top.values))
print(top_dict)
for comedian, top_words in top_dict.items():
    print(comedian)
    print(', '.join([word for word, count in top_words[0:14]]))
    print('___')

words = []
for comedian in data.columns:
    top = [word for (word, count) in top_dict[comedian]]
    for t in top:
        words.append(t)
print(words)

Counter(words).most_common()

add_stop_words = [word for word ,count in Counter(words).most_common() if count > 6]
print(add_stop_words)

data_clean = pd.read_pickle('data_clean.pkl')
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.transcript)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

pickle.dump(cv,open("cv_stop.pkl","wb"))
data_stop.to_pickle("dtm_stop.pkl")

wc = WordCloud(stopwords=stop_words,background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)

plt.rcParams['figure.figsize'] = [16,6]

full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']

for index , comedian in enumerate(data.columns):
    wc.generate(data_clean.transcript[comedian])

    plt.subplot(3,4, index+1)
    plt.imshow(wc,interpolation="bilinear")
    plt.axis("off")
    plt.title(full_names[index])
plt.show()

unique_list =[]
for comedian in data.columns:
    uniques = data[comedian].to_numpy().nonzero()[0].size
    unique_list.append(uniques)

data_words = pd.DataFrame(list(zip(full_names,unique_list)), columns=['comedian','unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')
print(data_unique_sort)

totals_list=[]
for comedian in data.columns:
    totals = sum(data[comedian])
    totals_list.append(totals)

run_times = [60, 59, 80, 60, 67, 73, 77, 63, 62, 58, 76, 79]

data_words['total_words'] = totals_list
data_words['run_times'] = run_times
data_words['words_per_minute'] = data_words['total_words']/data_words['run_times']

data_wpm_sort = data_words.sort_values(by='words_per_minute')
print(data_wpm_sort)

y_pos = np.arange(len(data_words))


plt.subplot(1, 2, 1)
plt.barh(y_pos, data_unique_sort.unique_words, align='center')
plt.yticks(y_pos, data_unique_sort.comedian)
plt.title('Number of Unique Words', fontsize=20)

plt.subplot(1, 2, 2)
plt.barh(y_pos, data_wpm_sort.words_per_minute, align='center')
plt.yticks(y_pos, data_wpm_sort.comedian)
plt.title('Number of Words Per Minute', fontsize=20)

plt.tight_layout()
plt.show()

Counter(words).most_common()

data_bad_words = data.transpose()[['fucking', 'fuck', 'shit']]
data_profanity = pd.concat([data_bad_words.fucking + data_bad_words.fuck, data_bad_words.shit], axis=1)
data_profanity.columns = ['f_word', 's_word']
print(data_profanity)

plt.rcParams['figure.figsize'] = [10,8]
for i, comedian in enumerate(data_profanity.index):
    x = data_profanity.f_word.loc[comedian]
    y = data_profanity.s_word.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x + 1.5, y + 0.5, full_names[i], fontsize=10)
    plt.xlim(-5, 155)

plt.title('Number of Bad Words Used in Routine', fontsize=20)
plt.xlabel('Number of F Bombs', fontsize=15)
plt.ylabel('Number of S Words', fontsize=15)

plt.show()