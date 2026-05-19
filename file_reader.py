import pandas as pd
from nltk.tokenize import sent_tokenize

class FileReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self):
        messages_data = pd.read_csv(self.file_path)
        
        chunks = []
        unique_id = []

        if messages_data.shape[1] != 2:
            print("Error: The CSV file should have 2 columns : unique id, message.")
        else:
            for row in messages_data.itertuples(index=False):
                message_id = row[0]
                message_text = str(row[1]) if pd.notna(row[1]) else ""
                
                # Split the message into sentences 
                sentences = sent_tokenize(message_text)
                
                for sentence in sentences:
                    if sentence.strip():
                        chunks.append(sentence)
                        unique_id.append(message_id)
        return unique_id, chunks
                        