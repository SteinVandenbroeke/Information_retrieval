import math
import os
import re
from collections import defaultdict
import numpy as np
import datetime
import sys

from numpy.ma.core import array


def tokenize_documents(path_to_documents):
    start_time = datetime.datetime.now()
    tokecounter = defaultdict(lambda: np.array([0, 0]))
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
                        inverted_index[x][tokecounter[x][1]] = doc_id
                    else:
                        array = np.empty(tokecounter[x][0])
                        array[0] = doc_id
                        inverted_index[x] = array

                    tokecounter[x][1] += 1

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
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_amount = len(os.listdir(path_to_documents))-2 #2 non txt files in database
    print(doc_amount, " documents found.")
    counter = 0
    for d in os.listdir(path_to_documents):
        if d.endswith('.txt'):
            doc_id = int(d.replace("output_", "").replace(".txt", ""))
            file_path = os.path.join(path_to_documents, d)
            with open(file_path, 'r') as file:
                terms = process_query(file.read())
                #terms = re.sub(r'\W+', ' ', file.read().lower()).split()
                for x in terms:
                    inverted_index[x][doc_id] += 1
        if (counter % 10000 == 0):
            print("done: ", counter)
        counter += 1
    print("done indexing")
    return inverted_index, doc_amount

def process_query(query):
    terms = query.lower().split()
    #terms = re.findall(r'\w+', query.lower())
    # todo eventueel terms als "het" en "en" eruit halen
    return terms

def retrieve_documents(query_terms, inverted_index, doc_amount):
    doc_query_vector = defaultdict(list)
    query_vector_total_pow = 0
    doc_vector_total_pow = defaultdict(float)
    print(set(query_terms))
    for i, term in enumerate(set(query_terms)):
        if not term in inverted_index: # skip if term not found in index
            continue
        dft = len(inverted_index[term])  # document frequency: number of documents that t occurs in
        N = doc_amount # Total amount of docs
        tftq = query_terms.count(term) # term frequency in a query
        idf_weight = np.log10(doc_amount / dft)  # document frequency weight
        query_value = ((1 + np.log10(tftq)) * idf_weight)
        for doc, tf in inverted_index[term].items():
            tftd = inverted_index[term][doc]# term frequency in a document
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
    return [doc[0] for doc in sorted(doc_scores, key=lambda x: x[1], reverse=True)]