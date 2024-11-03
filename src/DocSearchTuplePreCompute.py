import math
import os
from collections import defaultdict
from functools import partial
from itertools import chain
from threading import Thread
from operator import methodcaller
import numpy
import numpy as np
import datetime
import sys
import pickle
from joblib import dump, load
from numpy.ma.core import array
from heapq import heappush, nlargest
from src.Query_processors.QueryProcessor import QueryProcessor
from itertools import islice

def split_list(lst, n):
    # Calculate the size of each chunk
    avg = len(lst) / float(n)
    # Split the list
    return [lst[int(i * avg): int((i + 1) * avg)] for i in range(n)]

class DocSearchTuplePreCompute:
    def __init__(self, query_processor: QueryProcessor, folder_path: str):
        self.inverted_index = None
        self.query_processor = query_processor
        self.folder_path = folder_path
        self.doc_amount = 0

    def tokenize_documents(self):
        self.inverted_index, self.doc_amount = self._tokenize_documents(self.folder_path)

    def _tokenize_documents(self, path_to_documents):
        documentList = os.listdir(path_to_documents)
        documentList = sorted(documentList, key=lambda doc: int(doc.replace("output_", "").replace(".txt", "")) if doc.startswith("output_") and doc.endswith(".txt") else float('inf')) #sort doc id's

        start_time = datetime.datetime.now()
        tokecounter = defaultdict(int)
        counter = 0
        doc_amount = len(os.listdir(self.folder_path)) - 2
        print(f"tokenize {doc_amount} documents")
        for d in documentList:
            if d.endswith('.txt'):
                file_path = os.path.join(self.folder_path, d)
                with open(file_path, 'r') as file:
                    file_terms = self.query_processor.tokenize(file.read())
                    for x in set(file_terms):
                        tokecounter[x] += 1
            if (counter % 10000 == 0):
                print(counter, " | size: ", sys.getsizeof(tokecounter), " | items", len(tokecounter))
            counter += 1

        inverted_index = defaultdict()
        counter = 0
        for d in documentList:
            if d.endswith('.txt'):
                doc_id = int(d.replace("output_", "").replace(".txt", ""))
                file_path = os.path.join(self.folder_path, d)
                with open(file_path, 'r') as file:
                    fileContent = file.read()
                    file_terms = self.query_processor.tokenize(fileContent)
                    for term in file_terms:
                        if term in inverted_index:
                            index = np.searchsorted(inverted_index[term][0:,0], int(doc_id))
                            inverted_index[term][index] = (int(doc_id), inverted_index[term][index][1] + 1)
                        else:
                            array = np.empty((tokecounter[term], 2), dtype=float)
                            array[:] = [np.inf, 0]
                            array[0] = [int(doc_id), 1]
                            inverted_index[term] = array
                            del tokecounter[term]

            if(counter % 10000 == 0):
                print(counter, " | size: ",  sys.getsizeof(inverted_index), " | items", len(inverted_index))
            counter += 1

        total_size = sys.getsizeof(inverted_index)  # Size of the dictionary itself
        for key, value_set in inverted_index.items():
            total_size += sys.getsizeof(value_set)  # Add the size of each set
        counter = 0

        #precomputations
        for term, docid_amount in inverted_index.items():
            dft = docid_amount.shape[0]  # document frequency: number of documents that t occurs in
            idf_weight = np.log10(doc_amount / dft)
            # Combine them into a structured array with two columns
            for i, (doc_id, count) in enumerate(docid_amount):
                tftd = int(count)  # term frequency in a document
                tf_weight = (1 + np.log10(int(tftd))) if int(tftd) > 0 else 0  # term frequency weight for document d
                inverted_index[term][i][1] = float(tf_weight * idf_weight)
            if (counter % 10000 == 0):
                print("Tupelize data", counter)
            counter += 1

        print("total size: ", total_size)
        print("done indexing")
        end_time = datetime.datetime.now()
        print("time measurement: ", end_time - start_time)
        
        return inverted_index, doc_amount

# source = https://stackoverflow.com/questions/22878743/how-to-split-dictionary-into-multiple-dictionaries-fast
    def chunks(self,data, SIZE=50000):
        it = iter(data)
        for i in range(0, len(data), SIZE):
            yield {k: data[k] for k in islice(it, SIZE)}

    def saveInverseIndex(self, savedFile):
        for i, item in enumerate(self.chunks(self.inverted_index),1):
            dump(self.inverted_index, f"preprocessing/{savedFile}_chuck_{i}" + '.joblib')
            print("dump", i)

    def openPresaved(self, savedFile):
        preprocedFiles = os.listdir("preprocessing/")
        self.inverted_index = defaultdict()
        self.doc_amount = len(os.listdir(self.folder_path)) - 2
        for file in preprocedFiles:
            print(file)
            if file.endswith('.joblib'):
                self.inverted_index = self.inverted_index | load(f"preprocessing/{file}")

    def retrieve_documents(self, query, itemAmount):
        query_terms = self.query_processor.tokenize(query)
        doc_query_vector = defaultdict(float)
        doc_vector_total_pow = defaultdict(float)
        for i, term in enumerate(set(query_terms)):
            if not term in self.inverted_index:  # skip if term not found in index
                continue
            dft = len(self.inverted_index[term])  # document frequency: number of documents that t occurs in
            tftq = query_terms.count(term)  # term frequency in a query
            idf_weight = np.log10(self.doc_amount / dft)  # document frequency weight
            query_value = ((1 + np.log10(tftq)) * idf_weight)
            for doc_id, doc_value in self.inverted_index[term]:
                doc_query_vector[doc_id] += doc_value * query_value
                doc_vector_total_pow[doc_id] += doc_value ** 2

        doc_scores = []
        for doc in doc_query_vector:
            doc_norm = np.sqrt(doc_vector_total_pow[doc])
            if doc_norm > 0:
                normalized_score = doc_query_vector[doc] / doc_norm  # normalize
                heappush(doc_scores, (normalized_score, int(doc)))

        returnList = [doc[1] for doc in nlargest(itemAmount, doc_scores)]
        return returnList