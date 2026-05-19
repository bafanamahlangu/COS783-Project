from keyword_searcher import KeywordSearcher

if __name__ == "__main__":
    file_name = input("Enter the messages csv file name: ")
    file_path = f"Data/{file_name}"

    keywords_input = input("Enter the keywords to search: ")

    top_5_indeces, top_5_scores = KeywordSearcher.search_keywords(keywords_input, file_path)

    #loop through the top 5 indeces and print the message id, matched sentenced and score

    