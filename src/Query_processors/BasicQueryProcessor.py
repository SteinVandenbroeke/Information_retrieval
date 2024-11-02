from src.Query_processors.QueryProcessor import QueryProcessor


class BasicQueryProcessor(QueryProcessor):
    def __init__(self):
        pass

    def tokenize(self, query):
        tokens = query.lower().split()
        #terms = re.findall(r'\w+', query.lower())
        # todo eventueel terms als "het" en "en" eruit halen
        return tokens

