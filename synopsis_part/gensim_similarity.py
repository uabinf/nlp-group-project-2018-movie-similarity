import glob
import gensim
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

tf_idf = gensim.models.TfidfModel(corpus)
#print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
#print(s)

sims = gensim.similarities.Similarity(".",tf_idf[corpus], num_features=len(dictionary))
print(sims)
print(type(sims))


#print(raw_documents[7])
query_doc = [w.lower() for w in word_tokenize(raw_documents[2])]
#print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
#print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
#print(query_doc_tf_idf)

print("***************")
print(sims[query_doc_tf_idf])