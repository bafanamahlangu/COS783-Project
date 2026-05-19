from file_reader import FileReader
from keyword_searcher import KeywordSearcher, extract_entities

if __name__ == "__main__":
    file_name = input("Enter the messages csv file name: ")
    file_path = f"Data/{file_name}"

    keywords_input = input("Enter the keywords to search: ")

    top_indices, top_scores = KeywordSearcher.search_keywords(keywords_input, file_path)

    if not top_indices:
        print("No results found.")
    else:
        # Re-load chunks so we can display matched text alongside scores
        _, chunks = FileReader(file_path).read_file()
        message_ids, _ = FileReader(file_path).read_file()

        print(f"\nTop {len(top_indices)} results for: '{keywords_input}'\n" + "-" * 50)
        for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
            chunk = chunks[idx]
            msg_id = message_ids[idx]
            entities = extract_entities(chunk)

            print(f"[{rank}] Score: {score:.4f} | Message ID: {msg_id}")
            print(f"    Text: {chunk}")
            if entities:
                print(f"    Entities: {entities}")
            print()
