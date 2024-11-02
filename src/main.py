import os
import sys
from collections import defaultdict
from functools import partial
from itertools import chain
from operator import methodcaller
from threading import Thread
import datetime

import numpy as np

from src.DocSearch import DocSearch
from src.DocSearchDictPreCompute import DocSearchDictPreCompute
from src.DocSearchTuplePreCompute import DocSearchTuplePreCompute
from src.Query_processors.AdvancedQueryProcessor import AdvancedQueryProcessor
from src.Query_processors.BasicQueryProcessor import BasicQueryProcessor
from src.Query_processors.QueryProcessor import QueryProcessor
from src.test_queries import test_queries

if __name__ == '__main__':
    # query_processor = AdvancedQueryProcessor()
    # docsearch = DocSearchTuplePreCompute(query_processor, "./../../datasets/full_docs_small")
    # docsearch.tokenize_documents()
    # docsearch.saveInverseIndex("small_export")

    start_time = datetime.datetime.now()
    query_processor = AdvancedQueryProcessor()
    docsearch = DocSearchTuplePreCompute(query_processor, "./../../datasets/full_docs")
    docsearch.tokenize_documents()
    #docsearch.openPresaved("small_export")

    end_time_doc = datetime.datetime.now()
    print("Doc processing: ", end_time_doc - start_time)
    docsearch.saveInverseIndex("large_export")
    print("export done")
    #test_queries(docsearch, True)






    # query_processor = AdvancedQueryProcessor()
    # docsearch = DocSearchTuplePreCompute(query_processor, "./../../datasets/full_docs_small")
    # docsearch.tokenize_documents()
    #
    # docsearch1 = DocSearchDictPreCompute(query_processor, "./../../datasets/full_docs_small")
    # docsearch1.tokenize_documents()
    #
    # for term, tupels in docsearch.inverted_index.items():
    #     for doc_id, calcValue in tupels:
    #         if calcValue != docsearch1.inverted_index[term][int(doc_id)] and term == "the":
    #             print("FOUT", term, int(doc_id), calcValue, docsearch1.inverted_index[term][doc_id])


# def split_list(lst, n):
#     # Calculate the size of each chunk
#     avg = len(lst) / float(n)
#     # Split the list
#     return [lst[int(i * avg): int((i + 1) * avg)] for i in range(n)]
#
# if __name__ == '__main__':
#     #tokenize_documents("./../../datasets/full_docs/")
#     # start_time = datetime.datetime.now()
#     # indexPath = "./../../datasets/full_docs/"
#     # documentList = os.listdir("./../../datasets/full_docs")
#     # AMOUNT_OF_THREADS = 20
#     # documentLists = split_list(documentList, AMOUNT_OF_THREADS)
#     # threads = [None] * AMOUNT_OF_THREADS
#     # results = [None] * AMOUNT_OF_THREADS
#     # for i, documentListChunck in enumerate(documentLists):
#     #     print("start new thread")
#     #     threads[i] = Thread(target=tokenize_documents, args=["./../../datasets/full_docs", documentListChunck, results, i])
#     #     threads[i].start()
#     #
#     # print("joining threads")
#     # for t in threads:
#     #     t.join()
#     #
#     # print("joined threads")
#     # print("merging thread results")
#     #
#     # #mergin results
#     # inverted_index = defaultdict(partial(np.ndarray, 0, dtype=int))
#     # dict_items = map(methodcaller('items'), (item[0] for item in results))
#     # for k, v in chain.from_iterable(dict_items):
#     #     inverted_index[k] = np.concatenate([inverted_index[k], v])
#     #
#     # print("all thread results merged")
#     # end_time = datetime.datetime.now()
#     # print("time measurement: ", end_time - start_time)
#     # total_size = sys.getsizeof(inverted_index)  # Size of the dictionary itself
#     # for key, value_set in inverted_index.items():
#     #     total_size += sys.getsizeof(value_set)  # Add the size of each set
#     # print("total size: ", total_size)
#
#     non_multithreaded = tokenize_documents("./../../datasets/full_docs", saveToDisk="fulldataset")
#     #print(non_multithreaded == inverted_index)
#
#     #inverted_index = openPresaved("large")
#     #index_documents("./../../datasets/full_docs_small/")
#     # example test from slides
#     # inverted_index = {'car': {'1':27, '2':4, '3':24}, 'auto': {'1':3, '2':33}, 'insurance': {'2':33, '3':29}, 'best': {'1':14, '3':17}}
#     # doc_amount = 3
#     # query = "best auto insurance"
#     #
#     #
#     # # inverted_index, doc_amount = index_documents('../datasets/full_docs_small')
#     # # query = "types of road hugger tires"
#     # # print(query)
#     # # query_terms = process_query(query)
#     # # result = retrieve_documents(query_terms, inverted_index, doc_amount)
#     # # print(result)
#     #
#     # test_queries(False)