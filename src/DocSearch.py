import math
import os
from collections import defaultdict
import numpy as np
import datetime
import sys
import pickle
from joblib import dump, load
from numpy.ma.core import array

from src.Query_processors.QueryProcessor import QueryProcessor


class DocSearch:
    def __init__(self, query_processor: QueryProcessor, folder_path: str):
        self.inverted_index = None
        self.query_processor = query_processor
        self.folder_path = folder_path
        self.doc_amount = 0

    def tokenize_documents(self):
        self.inverted_index, self.doc_amount = self._tokenize_documents()

    def _tokenize_documents(self, documentList = None, result = None, i = 0):
        if documentList == None:
            documentList = os.listdir(self.folder_path)
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
                    for x in file_terms:
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
                    for x in file_terms:
                        if x in inverted_index:
                            inverted_index[x][0] += int(1)
                            inverted_index[x][inverted_index[x][0]] = doc_id
                        else:
                            array = np.empty(tokecounter[x] + 1, dtype=int)
                            array[0] = int(1)
                            array[1] = doc_id
                            inverted_index[x] = array
                            del tokecounter[x]

            if(counter % 10000 == 0):
                print(counter, " | size: ",  sys.getsizeof(inverted_index), " | items", len(inverted_index))
            counter += 1

        total_size = sys.getsizeof(inverted_index)  # Size of the dictionary itself
        for key, value_set in inverted_index.items():
            total_size += sys.getsizeof(value_set)  # Add the size of each set

        #remove temp indexes
        for key, value in inverted_index.items():
            inverted_index[key] = value[1:]

        if result != None:
            result[i] = (inverted_index, doc_amount)

        print("total size: ", total_size)
        print("done indexing")
        end_time = datetime.datetime.now()
        print("time measurement: ", end_time - start_time)
        
        return inverted_index, doc_amount

    def openPresaved(self, savedFile):
        return load(savedFile + '.joblib')

    def _index_documents(self, path_to_documents):
        print("index_documents")
        start_time = datetime.datetime.now()
        inverted_index = defaultdict(lambda: defaultdict(int))
        doc_amount = len(os.listdir(path_to_documents))-2 #2 non txt files in database
        print(doc_amount, " documents found.")
        counter = 0
        for d in os.listdir(path_to_documents):
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
        print("done indexing")
        end_time = datetime.datetime.now()
        print("time measurement: ", end_time - start_time)
        total_size = sys.getsizeof(inverted_index)  # Size of the dictionary itself
        for key, value_set in inverted_index.items():
            total_size += sys.getsizeof(value_set)  # Add the size of each set
        print("total size: ", total_size)

        return inverted_index, doc_amount

    def retrieve_documents(self, query_terms, itemAmount):
        doc_query_vector = defaultdict(list)
        query_vector_total_pow = 0
        doc_vector_total_pow = defaultdict(float)
        print(set(query_terms))
        for i, term in enumerate(set(query_terms)):
            if not term in self.inverted_index: # skip if term not found in index
                continue
            dft = len(set(self.inverted_index[term]))  # document frequency: number of documents that t occurs in
            N = self.doc_amount # Total amount of docs
            tftq = query_terms.count(term) # term frequency in a query
            idf_weight = np.log10(self.doc_amount / dft)  # document frequency weight
            query_value = ((1 + np.log10(tftq)) * idf_weight)
            print(set(self.inverted_index[term]))
            for doc in set(self.inverted_index[term]):
                tftd = np.count_nonzero(self.inverted_index[term] == doc)# term frequency in a document
                tf_weight = (1 + np.log10(tftd)) if tftd > 0 else 0  # term frequency weight for document d
                doc_query_vector[doc].append(((tf_weight * idf_weight) * query_value))
                #print(term, doc, ": ", doc_scores[doc])
                doc_vector_total_pow[doc] += math.pow(tf_weight * idf_weight, 2)

            query_vector_total_pow += math.pow(query_value, 2)

        query_norm = np.sqrt(query_vector_total_pow)
        doc_scores = []
        for doc in doc_query_vector:
            doc_norm = np.sqrt(doc_vector_total_pow[doc])
            if doc_norm > 0:
                doc_query_vector[doc] = [(v/(doc_norm * query_norm)) for v in doc_query_vector[doc]] # normalize
                doc_scores.append((doc, sum(x for x in doc_query_vector[doc])))
                #print(doc, ": ", doc_scores[doc])
        return [doc[0] for doc in sorted(doc_scores, key=lambda x: x[1], reverse=True)[0:itemAmount]]