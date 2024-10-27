import os
import re
from collections import defaultdict
import numpy as np

def index_documents(path_to_documents):
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_amount = len(os.listdir(path_to_documents))-2 #2 non txt files in database
    print(doc_amount, " documents found.")
    for d in os.listdir(path_to_documents):
        if d.endswith('.txt'):
            doc_id = int(d.replace("output_", "").replace(".txt", ""))
            file_path = os.path.join(path_to_documents, d)
            with open(file_path, 'r') as file:
                terms = re.sub(r'\W+', ' ', file.read().lower()).split()
                for x in terms:
                    inverted_index[x][doc_id] += 1
    print("done indexing")
    return inverted_index, doc_amount

def process_query(query):
    terms = re.findall(r'\w+', query.lower())
    # todo eventueel terms als "het" en "en" eruit halen + al checken of terms in index zitten, en anders wegdoen
    return terms

def retrieve_documents(query_terms, inverted_index, doc_amount):
    query_vector = []
    doc_scores = defaultdict(int)
    for i, term in enumerate(query_terms):
        if not term in inverted_index: # skip if term not found in index
            continue
        query_vector.append(np.log10(doc_amount/len(inverted_index[term])))
        df = len(inverted_index[term])  # document frequency: number of documents that t occurs in
        idf_weight = np.log10(doc_amount / df)  # document frequency weight
        for doc, tf in inverted_index[term].items():
            tf_weight = 1 + np.log10(inverted_index[term][doc]) if inverted_index[term][doc] > 0 else 0  # term frequency weight for document d
            doc_scores[doc] += (tf_weight * idf_weight) * query_vector[i]
            #print(term, doc, ": ", doc_scores[doc])
    query_norm = np.sqrt(sum(v ** 2 for v in query_vector))
    for doc in doc_scores:
        doc_vector = []
        for term in query_terms:
            if doc in inverted_index[term]:
                doc_vector.append((1 + np.log10(inverted_index[term][doc])) * np.log10(doc_amount / len(inverted_index[term])))
            else:
                doc_vector.append(0)
        doc_norm = np.sqrt(sum(v ** 2 for v in doc_vector))
        if doc_norm > 0:
            doc_scores[doc] /= (query_norm * doc_norm) # normalize
            #print(doc, ": ", doc_scores[doc])
    return sorted(doc_scores, key=doc_scores.get, reverse=True)