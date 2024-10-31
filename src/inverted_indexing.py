import math
import os
import re
from collections import defaultdict
import numpy as np
import datetime
import sys

from numpy.ma.core import array


def tokenize_documents(path_to_documents):
    tokecounter = defaultdict(lambda: np.array([0, 0]))
    start_time = datetime.datetime.now()

    counter = 0
    print("start")
    for d in os.listdir(path_to_documents):
        if d.endswith('.txt'):
            file_path = os.path.join(path_to_documents, d)
            with open(file_path, 'r') as file:
                terms = file.read().lower().split()
                for x in terms:
                    tokecounter[x][0] += 1
        if (counter % 10000 == 0):
            print(counter, " | size: ", sys.getsizeof(tokecounter), " | items", len(tokecounter))
        counter += 1

    print("token counter speedtest")
    inverted_index = defaultdict()
    doc_amount = len(os.listdir(path_to_documents))
    print(doc_amount, " documents found.")
    counter = 0
    for d in os.listdir(path_to_documents):
        if d.endswith('.txt'):
            doc_id = int(d.replace("output_", "").replace(".txt", ""))
            file_path = os.path.join(path_to_documents, d)
            with open(file_path, 'r') as file:
                terms = file.read().lower().split()
                for x in terms:
                    if x in inverted_index:
                        empty_index = np.where(np.isnan(tokecounter))[0][0] if np.any(np.isnan(tokecounter)) else None

                        inverted_index[x][empty_index] = doc_id

                    else:
                        array = np.empty(tokecounter[x][0])
                        array[0] = doc_id
                        del tokecounter[x]
                        inverted_index[x] = array

        if(counter % 10000 == 0):
            print(counter, " | size: ",  sys.getsizeof(inverted_index), " | items", len(inverted_index))
        counter += 1

    total_size = sys.getsizeof(inverted_index)  # Size of the dictionary itself
    for key, value_set in inverted_index.items():
        total_size += sys.getsizeof(value_set)  # Add the size of each set
    print("total size: ", total_size)
    print("done indexing")
    end_time = datetime.datetime.now()
    print("time measurement: ", end_time - start_time)
    print(inverted_index)
    return inverted_index, doc_amount


def index_documents(path_to_documents):
    counter = 0
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_amount = len(os.listdir(path_to_documents))-2 #2 non txt files in database
    print(doc_amount, " documents found.")
    for i, d in enumerate(os.listdir(path_to_documents)):
        if d.endswith('.txt'):
            doc_id = int(d.replace("output_", "").replace(".txt", ""))
            file_path = os.path.join(path_to_documents, d)
            with open(file_path, 'r') as file:
                #terms = file.read().lower().split()
                terms = process_query(file.read())
                for x in terms:
                    inverted_index[x][doc_id] += 1
            if(counter%10000 == 0):
                print("done: ", counter)
            counter += 1
    print("done indexing")
    return inverted_index, math.floor(doc_amount/2)

def process_query(query):
    terms = query.lower().split()
    #terms = re.findall(r'\w+', query.lower())
    # todo eventueel terms als "het" en "en" eruit halen
    return terms

def retrieve_documents(query_terms, inverted_index, doc_amount):
    query_vector = []
    doc_vectors = defaultdict(lambda: defaultdict(float))
    for i, term in enumerate(query_terms):
        if not term in inverted_index: # skip if term not found in index
            query_vector.append(0)
            continue
        query_vector.append((1 + np.log10(query_terms.count(term))) * np.log10(doc_amount/len(inverted_index[term])))
        df = len(inverted_index[term])  # document frequency: number of documents that t occurs in
        idf_weight = np.log10(doc_amount / df)  # document frequency weight
        for doc, tf in inverted_index[term].items():
            tf_weight = (1 + np.log10(tf)) if inverted_index[term][doc] > 0 else 0  # term frequency weight for document d
            doc_vectors[doc][term] = (tf_weight * idf_weight)
    query_norm = np.sqrt(sum(math.pow(v,2) for v in query_vector))
    doc_scores = defaultdict(int)
    for doc in doc_vectors:
        doc_vector = doc_vectors[doc]
        doc_norm = np.sqrt(sum(math.pow(v,2)for v in doc_vector.values()))
        for i, term in enumerate(query_terms):
            doc_vector[term] *= (query_vector[i] / (doc_norm * query_norm))
        doc_scores[doc] =  sum(x for x in doc_vectors[doc].values())
    #print(doc_scores)
    return [int(doc) for doc in sorted(doc_scores, key=doc_scores.get, reverse=True)]