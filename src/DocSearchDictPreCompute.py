import math
import os
from collections import defaultdict
import numpy as np
import datetime
import sys
import pickle
from joblib import dump, load
from numpy.ma.core import array
from sortedcontainers import SortedList
from heapq import heappush, nlargest

from src.Query_processors.QueryProcessor import QueryProcessor


class DocSearchDictPreCompute:
    def __init__(self, query_processor: QueryProcessor, folder_path: str):
        self.inverted_index = None
        self.query_processor = query_processor
        self.folder_path = folder_path
        self.doc_amount = 0

    def tokenize_documents(self):
        self.inverted_index, self.doc_amount = self._index_documents(self.folder_path)

    def openPresaved(self, savedFile):
        return load(savedFile + '.joblib')

    def _index_documents(self, path_to_documents):
        documentList = os.listdir(self.folder_path)
        documentList = sorted(documentList,
                              key=lambda doc: int(doc.replace("output_", "").replace(".txt", "")) if doc.startswith(
                                  "output_") and doc.endswith(".txt") else float('inf'))
        print("index_documents")
        start_time = datetime.datetime.now()
        inverted_index = defaultdict(lambda: defaultdict(float))
        doc_amount = len(os.listdir(path_to_documents))-2 #2 non txt files in database
        print(doc_amount, " documents found.")
        counter = 0

        for d in documentList:
            if d.endswith('.txt'):
                doc_id = int(d.replace("output_", "").replace(".txt", ""))
                file_path = os.path.join(path_to_documents, d)
                with open(file_path, 'r') as file:
                    terms = self.query_processor.tokenize(file.read())
                    #terms = re.sub(r'\W+', ' ', file.read().lower()).split()
                    for x in terms:
                        inverted_index[x][doc_id] += 1
            if (counter % 10000 == 0):
                print("done: ", counter)
            counter += 1

        #precomputations
        for term in inverted_index:
            dft = len(inverted_index[term])  # document frequency: number of documents that t occurs in
            idf_weight = np.log10(doc_amount / dft)  # document frequency weight
            for doc in inverted_index[term]:
                tftd = inverted_index[term][doc]  # term frequency in a document
                tf_weight = (1 + np.log10(tftd)) if tftd > 0 else 0  # term frequency weight for document d
                inverted_index[term][doc] = (((tf_weight * idf_weight)))
                # print(term, doc, ": ", doc_scores[doc])

        print("done indexing")
        end_time = datetime.datetime.now()
        print("time measurement: ", end_time - start_time)
        total_size = sys.getsizeof(inverted_index)  # Size of the dictionary itself
        for key, value_set in inverted_index.items():
            total_size += sys.getsizeof(value_set)  # Add the size of each set
        print("total size: ", total_size)

        return inverted_index, doc_amount

    def retrieve_documents(self, query_terms, itemAmount):
        start_time = datetime.datetime.now()
        doc_query_vector = defaultdict(float)
        doc_vector_total_pow = defaultdict(float)
        print(set(query_terms))
        for i, term in enumerate(set(query_terms)):
            if not term in self.inverted_index:  # skip if term not found in index
                continue
            dft = len(self.inverted_index[term])  # document frequency: number of documents that t occurs in
            tftq = query_terms.count(term)  # term frequency in a query
            idf_weight = np.log10(self.doc_amount / dft)  # document frequency weight
            query_value = ((1 + np.log10(tftq)) * idf_weight)
            for doc_id, doc_value in self.inverted_index[term].items():
                doc_query_vector[doc_id] += ((doc_value * query_value))
                doc_vector_total_pow[doc_id] += doc_value ** 2

        end_time_first_loop = datetime.datetime.now()

        doc_scores = []
        for doc in doc_query_vector:
            doc_norm = np.sqrt(doc_vector_total_pow[doc])
            if doc_norm > 0:
                normalized_score = doc_query_vector[doc] / doc_norm  # normalize
                heappush(doc_scores, (normalized_score, doc))
                # print(doc, ": ", doc_scores[doc])

        end_time_second_loop = datetime.datetime.now()

        returnList = [doc[1] for doc in nlargest(itemAmount, doc_scores)]
        end_time= datetime.datetime.now()

        print(f"first loop time: {end_time_first_loop - start_time} | second loop time {end_time_second_loop - end_time_first_loop} | list processing time {end_time-end_time_second_loop} | total time: {end_time - start_time}")

        return returnList