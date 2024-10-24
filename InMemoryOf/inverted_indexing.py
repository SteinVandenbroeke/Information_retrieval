import os
import re


def tokenize_documents(path_to_documents):
    terms = {}
    print(len(os.listdir(path_to_documents)), " documents found.")
    for d in os.listdir(path_to_documents):
        if d.endswith('.txt') and os.path.isfile(os.path.join(path_to_documents, d)):
            doc_id = int(d.replace("output_", "").replace(".txt", ""))
            file_path = os.path.join(path_to_documents, d)
            with open(file_path, 'r') as file:
                f = file.read()
                cleaned_text = re.sub(r'[^a-zA-Z0-9]', ' ', f)
                doc_tokens = cleaned_text.lower().split()
                for x in doc_tokens:
                       if x in terms.keys():
                           terms[x] = terms[x] + (doc_id,)
                       else:
                           terms[x] = (doc_id,)
                #terms += [(x, doc_id) for x in doc_tokens]
    print("done tokens")
    print(terms)
    print("done sort")


