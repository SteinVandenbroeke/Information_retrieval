import os
import re
from collections import defaultdict

def tokenize_documents(path_to_documents):
    inverted_index = defaultdict(set)
    print(len(os.listdir(path_to_documents)), " documents found.")
    for d in os.listdir(path_to_documents):
        if d.endswith('.txt'):
            doc_id = int(d.replace("output_", "").replace(".txt", ""))
            file_path = os.path.join(path_to_documents, d)
            with open(file_path, 'r') as file:
                terms = re.sub(r'\W+', ' ', file.read().lower()).split()
                for x in terms:
                    inverted_index[x].add(doc_id)
    print("done indexing")