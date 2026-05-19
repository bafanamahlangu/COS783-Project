from sentence_transformers import SentenceTransformer
from  file_reader import FileReader

class KeywordSearcher:
    def __init__(self, keywords, file_path):
        self.keywords = keywords
        self.file_path = file_path

    def search_keywords(keywords, file_path):
        file_reader = FileReader(file_path) 

        message_id, chunks = file_reader.read_file()

        #print(message_id) #testing
        #print(chunks)
        #print(keywords)

        #Implement sentence transformer embeddings for keywords and chunks and search logic using cosine similarity

        
        return #top 5 indeces and scores