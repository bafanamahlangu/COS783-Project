from file_reader import FileReader

# Test with the example CSV
file_reader = FileReader("Data\\example.csv")
unique_ids, chunks = file_reader.read_file()

print("Unique IDs:")
print(unique_ids)
print("\nChunks:")
print(chunks)
print(f"\nTotal chunks: {len(chunks)}")
print(f"Total unique IDs: {len(unique_ids)}")
