import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

corpus = []
for file in glob.glob("synopsis/*"):
    with open(file, "r") as paper:
        corpus.append((file, paper.read()))

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
tfidf_matrix =  tf.fit_transform([content for file, content in corpus])

print(len(corpus))


#for x in range(0,len(corpus)):
#	print(x,corpus[x][0])

print("Finding similar movies of:", corpus[7][0])
for index, score in find_similar(tfidf_matrix, 2):
       print(score, corpus[index][0]) 