import datetime

import pandas as pd

from src.DocSearch import DocSearch
from src.DocSearchDictPreCompute import DocSearchDictPreCompute
from src.DocSearchOld import DocSearchOld
from src.DocSearchTuplePreCompute import DocSearchTuplePreCompute
from src.Query_processors.AdvancedQueryProcessor import AdvancedQueryProcessor
from src.Query_processors.BasicQueryProcessor import BasicQueryProcessor


def test_queries(docsearch, small = True):
    # start_time = datetime.datetime.now()
    # pad = './../../datasets/full_docs'
    # if small:
    #     pad = pad.replace('full_docs', 'full_docs_small')
    # query_processor = AdvancedQueryProcessor()
    # docsearch = DocSearchTuplePreCompute(query_processor, pad)
    # docsearch.tokenize_documents()
    start_time = datetime.datetime.now()

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
        query_terms = docsearch.query_processor.tokenize(query)
        result = docsearch.retrieve_documents(query_terms, 20)
        query_result_doc = int(query_result_data[query_result_data['Query_number'] == query_id]['doc_number'].tolist()[0])

        #print("query: ", query, ", result: ", query_result_doc, ", you gave: ", result)
        if query_result_doc in result:
            passes += 1
            #print("passed")
        else:
            fails += 1
            #print("failed")
    end_time_query_test = datetime.datetime.now()
    print("Query time: ", end_time_query_test - start_time)
    print(passes, " out of ", passes+fails, " queries passed.")
