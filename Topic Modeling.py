import pandas as pd
import pickle
from gensim import matutils, models
import scipy.sparse
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_pickle('dtm_stop.pkl')
print(data)

tdm = data.transpose()
print(tdm.head())

sparse_count = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_count)

cv = pickle.load(open("cv_stop.pkl","rb"))
id2word = dict((v,k) for k,v in cv.vocabulary_.items())

lda =models.LdaModel(corpus=corpus,id2word=id2word, num_topics=3, passes = 10)
print(lda.print_topics())

def nouns(text):
    is_noun = lambda pos:pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
    return ' '.join(all_nouns)

data_clean = pd.read_pickle('data_clean.pkl')
print(data_clean)

data_nouns = pd.DataFrame(data_clean.transcript.apply(nouns))
print(data_nouns)

add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said']

stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

cvn = CountVectorizer(stop_words = stop_words)
data_cvn = cvn.fit_transform(data_nouns.transcript)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.index
print(data_dtmn)

corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))
id2word = dict((v,k) for k,v in cvn.vocabulary_.items())

ldan = models.LdaModel(corpus=corpusn, num_topics=3, id2word=id2word, passes=10)
print(ldan.print_topics())


def nouns_adj(text):
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)]
    return ' '.join(nouns_adj)

data_nouns_adj = pd.DataFrame(data_clean.transcript.apply(nouns_adj))
print(data_nouns_adj)

cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
data_cvna = cvna.fit_transform(data_nouns_adj.transcript)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index
print(data_dtmna)

corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())

ldana = models.LdaModel(corpus=corpusna, num_topics=2, id2word=id2wordna, passes=10)
print(ldana.print_topics())

ldana = models.LdaModel(corpus=corpusna, num_topics=3, id2word=id2wordna, passes=10)
print(ldana.print_topics())

ldana = models.LdaModel(corpus=corpusna, num_topics=4, id2word=id2wordna, passes=10)
print(ldana.print_topics())

ldana = models.LdaModel(corpus=corpusna, num_topics=4, id2word=id2wordna, passes=80)
print(ldana.print_topics())

corpus_transformed = ldana[corpusna]
list(zip([a for [(a,b)] in corpus_transformed], data_dtmna.index))