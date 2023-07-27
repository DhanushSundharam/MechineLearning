import stanza
import sys
import concurrent.futures

# Download the Tamil language model if it's not already downloaded
stanza.download('ta')

# Load the Tamil language model
nlp = stanza.Pipeline('ta')

def pos_tagging_tamil(sentence):
    # Process the input sentence
    doc = nlp(sentence)
    
    # Initialize an empty list to store tagged words and their POS
    tagged_words = []
    
    # Extract words and their POS from the processed document
    for sentence in doc.sentences:
        for word in sentence.words:
            tagged_words.append((word.text, word.upos))
    
    return tagged_words

# Sample Tamil sentences for POS tagging
tamil_sentences = [
    "தமிழ் மொழியின் அழகு அருவருக்கு புரிவது.",
    "எனக்கு தமிழ் மொழி பிடித்தது.",
    "அவர் மிகுந்த ஆச்சரியத்தை அடைந்தார்."
]

# Perform POS tagging on the Tamil sentences using multiprocessing
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(pos_tagging_tamil, tamil_sentences))

# Display the results with UTF-8 encoding
for i, tagged_words in enumerate(results):
    print(f"Sentence {i + 1}:")
    for word, pos in tagged_words:
        sys.stdout.buffer.write(f"{word} -> {pos}\n".encode('utf-8'))

# Save the tagged output to a file with UTF-8 encoding
with open('output.txt', 'w', encoding='utf-8') as f:
    for i, tagged_words in enumerate(results):
        f.write(f"Sentence {i + 1}:\n")
        for word, pos in tagged_words:
            f.write(f"{word} -> {pos}\n")
