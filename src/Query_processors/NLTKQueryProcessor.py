import os
import string
import re

from src.Query_processors.QueryProcessor import QueryProcessor
import nltk

class NLTKQueryProcessor(QueryProcessor):
    def __init__(self):
        self.query_removal_words = []
        if not os.path.isdir(os.path.expanduser('~/nltk_data')):
            nltk.download()

    def frequentWordIndexing(self, path_to_documents, index_precentage = 0.10, frequency_precentage = 0.01):
        documents = os.listdir(path_to_documents)
        total_docs = len(documents)
        stepSize = int(total_docs * index_precentage)
        all_tokens = []
        for i in range(0, len(documents), stepSize):
            if documents[i].endswith('.txt'):
                file_path = os.path.join(path_to_documents, documents[i])
                with open(file_path, 'r') as file:
                    tokens = self.tokenize(file.read())
                    all_tokens += tokens
        fdist = nltk.FreqDist(all_tokens)

        self.query_removal_words = [word for word, _ in fdist.most_common(int(frequency_precentage * len(fdist)))]
        print("Stop words", nltk.corpus.stopwords.words("english"))
        print(self.query_removal_words)


    def tokenize(self, query):
        tokens = re.findall(r'\w+', query.lower())

        stopwords = nltk.corpus.stopwords.words("english")

        # remove stopwords
        tokens = [token for token in tokens if token.lower() not in stopwords]
        #
        # lemmatizer = nltk.stem.WordNetLemmatizer()
        #
        # # lemmatize each token
        # tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens