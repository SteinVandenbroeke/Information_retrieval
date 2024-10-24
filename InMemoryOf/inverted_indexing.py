import os
import re
import sys
from collections import defaultdict
import datetime
import numpy as np

def tokenize_documents(path_to_documents):
    tokecounter = defaultdict(lambda: np.array([0, 0]))
    counter = 0
    print("start")
    for d in os.listdir(path_to_documents):
        if d.endswith('.txt'):
            file_path = os.path.join(path_to_documents, d)
            with open(file_path, 'r') as file:
                terms = re.sub(r'\W+', ' ', file.read().lower()).split()
                for x in terms:
                    tokecounter[x][0] += 1
        if (counter % 10000 == 0):
            print(counter, " | size: ", sys.getsizeof(tokecounter), " | items", len(tokecounter))
        counter += 1

    print("token counter speedtest")

    start_time = datetime.datetime.now()
    inverted_index = defaultdict()
    print(len(os.listdir(path_to_documents)), " documents found.")
    counter = 0
    for d in os.listdir(path_to_documents):
        if d.endswith('.txt'):
            doc_id = int(d.replace("output_", "").replace(".txt", ""))
            file_path = os.path.join(path_to_documents, d)
            with open(file_path, 'r') as file:
                terms = re.sub(r'\W+', ' ', file.read().lower()).split()
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