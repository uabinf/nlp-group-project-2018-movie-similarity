# Movie Similarity using Subtitles and Synopses 

1. Download subtitles (.srt) and place them into /srt folder
2. Preprocess subtitles
	- remove markup
	- tokenise
	- remove punctuation and stopwords
	- remove low info words
	- POS tag -> lemmatise 
3. Vectorise (bag-of-words corpus)
4. Remove low info words appearing in fewer than [i]N[/i] (=4) or greater than [i]M[/i] (=95%) films
5. Find optimal number of topics for LDA
6. Build LDA with optimal number of topics
7. Get topic distributions
8. Create wordclouds
9. Create a similarity matrix heatmap
