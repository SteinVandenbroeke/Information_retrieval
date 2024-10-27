from src.inverted_indexing import index_documents, process_query, retrieve_documents

if __name__ == '__main__':
    # example test from slides
    # inverted_index = {'car': {'1':27, '2':4, '3':24}, 'auto': {'1':3, '2':33}, 'insurance': {'2':33, '3':29}, 'best': {'1':14, '3':17}}
    # doc_amount = 3
    # query = "best auto insurance"
    inverted_index, doc_amount = index_documents('../datasets/full_docs_small')
    query = "types of road hugger tires"
    print(query)
    query_terms = process_query(query)
    result = retrieve_documents(query_terms, inverted_index, doc_amount)
    print(result)