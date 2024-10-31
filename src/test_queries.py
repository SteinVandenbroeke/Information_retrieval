from ctypes.wintypes import SMALL_RECT

from src.inverted_indexing import index_documents, process_query, retrieve_documents, tokenize_documents
import pandas as pd

def test_queries(small = True):
    inverted_index, doc_amount = tokenize_documents('../datasets/full_docs' + ('_small' if small else ''))
    print("Testing queries for ", "small" if small else "big", " database:")
    query_path = "../queries/dev_queries" + ("_small.csv" if small else ".csv")
    query_data = pd.read_csv(query_path, names=["Query number","Query"], skiprows=1)
    query_result_path = "../queries/dev_query_results" + ("_small.csv" if small else ".csv")
    query_result_data = pd.read_csv(query_result_path, names=["Query_number","doc_number"], skiprows=1)

    passes = 0
    fails = 0

    for q in query_data.values:
        query_id = q[0]
        query = q[1]
        query_terms = process_query(query)
        result = retrieve_documents(query_terms, inverted_index, doc_amount)[0]
        query_result_doc = int(query_result_data[query_result_data['Query_number'] == query_id]['doc_number'].tolist()[0])
        print("query: ", query, ", result: ", query_result_doc, ", you gave: ", result)
        if query_result_doc == result:
            passes += 1
            print("passed")
        else:
            fails += 1
            print("failed")
    print(passes, " out of ", passes+fails, " queries passed.")