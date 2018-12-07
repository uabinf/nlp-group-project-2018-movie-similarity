import glob
import gensim
import pandas as pd
from nltk.tokenize import word_tokenize

corpus = []
corpus_raw = []
for file in glob.glob("subset/*"):
    with open(file, "r") as paper:
        corpus.append((file, paper.read()))


for x in range(0,len(corpus)):
	print(x,corpus[x][0])

raw_documents = []
for x in range(0,len(corpus)):
	#print(corpus[x][1])
	raw_documents.append(corpus[x][1])

#print(raw_documents)


gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents]
#print(gen_docs)

dictionary = gensim.corpora.Dictionary(gen_docs)
#print(dictionary[5])
#print(dictionary.token2id['road'])
#print("Number of words in dictionary:",len(dictionary))
#for i in range(len(dictionary)):
#    print(i, dictionary[i])

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
#print(corpus)

tfidf_model = gensim.models.TfidfModel(corpus)
tfidf_corpus = tfidf_model[corpus]

total_topics = 40
lda_model_tfidf = gensim.models.LdaModel(corpus=tfidf_corpus, id2word=dictionary, chunksize = 2000, num_topics=total_topics, passes=200, iterations = 500, random_state = 40, eval_every=None)
lda_model_bow = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, chunksize = 2000, num_topics=total_topics, passes=200, iterations = 500, random_state = 40, eval_every=None)

query_doc = [w.lower() for w in word_tokenize(raw_documents[6])]
#print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
#print(query_doc_bow)
query_doc_tf_idf = tfidf_model[query_doc_bow]

query_doc2 = [w.lower() for w in word_tokenize(raw_documents[7])]
#print(query_doc)
query_doc_bow2 = dictionary.doc2bow(query_doc2)
#print(query_doc_bow)
query_doc_tf_idf2 = tfidf_model[query_doc_bow2]

lda_topics_2 = lda_model_bow[query_doc_bow]
lda_topics_2_tfidf = lda_model_tfidf[query_doc_tf_idf]
#print(lda_topics_2)
#print(lda_topics_2_tfidf)

lda_topics_5 = lda_model_bow[query_doc_bow2]
lda_topics_5_tfidf = lda_model_tfidf[query_doc_tf_idf2]
#print(lda_topics_5)
#print(lda_topics_5_tfidf)


df1 = pd.DataFrame(lda_topics_2_tfidf, columns=['topic', 'contrib'])
print(df1)

df2 = pd.DataFrame(lda_topics_5_tfidf, columns=['topic', 'contrib'])
print(df2)

simmilarity_2_5 = gensim.matutils.cossim(lda_topics_2_tfidf, lda_topics_5_tfidf)
print('similarity between docs 100 and 199', simmilarity_2_5)