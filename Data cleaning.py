import requests
import os
from bs4 import BeautifulSoup
import pickle
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer

# retrieving data from the website
def url_to_transcript(url):
    page = requests.get(url).text
    soup = BeautifulSoup(page,"lxml")
    text = [p.text for p in soup.find(class_ = "post-content").find_all('p')]
    print(url)
    return text

urls = ['http://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/11/dave-chappelle-age-spin-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/15/ricky-gervais-humanity-transcript/',
        'http://scrapsfromtheloft.com/2017/08/07/bo-burnham-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/05/24/bill-burr-im-sorry-feel-way-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/21/jim-jefferies-bare-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/02/john-mulaney-comeback-kid-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2017/10/21/hasan-minhaj-homecoming-king-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2017/09/19/ali-wong-baby-cobra-2016-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/03/anthony-jeselnik-thoughts-prayers-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/03/mike-birbiglia-my-girlfriends-boyfriend-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/19/joe-rogan-triggered-2016-full-transcript/']

comedians = ['louis','dave', 'ricky','bo','bill','jim','john','hasan','ali','anthony', 'mike','joe']

transcript = [url_to_transcript(u) for u in urls]

for i,c in enumerate(comedians):
    with open("transcript/"+c+".txt","wb") as file :
        pickle.dump(transcript[i],file)
data = {}
for i,c in enumerate(comedians):
    with open("transcript/"+c+".txt","rb") as file :
        data[c] = pickle.load(file)

print(data.keys())
print(data['louis'][:2])

print(next(iter(data.keys())))
print(next(iter(data.keys())))

def combine_text(list_of_text):
    combine_text = ' '.join(list_of_text)
    return combine_text

data_combined = {key: [combine_text(value)] for (key, value) in data.items()}

pd.set_option('max_colwidth',150)

data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()
print(data_df)

print(data_df.transcript.loc['ali'])

def clean_text_round1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)
data_clean = pd.DataFrame(data_df.transcript.apply(round1))
print(data_clean)

print(data_df)

full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']

data_df['full_name'] = full_names

data_df.to_pickle("corpus")

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.transcript)
data_dtm = pd.DataFrame(data_cv.toarray(),columns = cv.get_feature_names())
data_dtm.index = data_clean.index
print(data_dtm)

data_dtm.to_pickle("dtm.pkl")
data_clean.to_pickle("data_clean.pkl")
pickle.dump(cv,open("cv.pkl","wb"))