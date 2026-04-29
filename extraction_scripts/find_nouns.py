import csv
import re
from pathlib import Path
from collections import Counter

ROOT_DIR = Path("/home/us80abag/whisper_output")
OUTPUT_DIR = ROOT_DIR / "transcript_stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MAIN = OUTPUT_DIR / "proper_nouns_mid_sentence.csv"
OUTPUT_START = OUTPUT_DIR / "proper_nouns_sentence_start.csv"

NO_SPEECH_TOKEN = "<no speech detected>"

# Sentence splitter
SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+')

# Word tokenizer: letters + numbers + apostrophes + hyphens + slashes
WORD_REGEX = re.compile(r"\b[A-Za-z][A-Za-z0-9'/-]*\b")

def extract_capitalized_sequences(words):
    sequences = []
    current = []

    for word in words:
        if word and word[0].isupper():
            current.append(word)
        else:
            if current:
                sequences.append(" ".join(current))
                current = []
    if current:
        sequences.append(" ".join(current))

    return sequences

def main():
    transcript_files = sorted(ROOT_DIR.glob("*/transcripts/*.txt"))

    mid_sentence_counter = Counter()
    start_sentence_counter = Counter()

    for file_path in transcript_files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if text.lower() == NO_SPEECH_TOKEN:
                continue

            sentences = SENTENCE_SPLIT_REGEX.split(text)

            for sentence in sentences:
                words = WORD_REGEX.findall(sentence)
                if not words:
                    continue

                first_word = words[0]
                sequences = extract_capitalized_sequences(words)

                for seq in sequences:
                    seq_words = seq.split()
                    if seq_words[0] == first_word:
                        start_sentence_counter[seq] += 1
                    else:
                        mid_sentence_counter[seq] += 1

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Write mid-sentence CSV sorted by frequency descending
    with open(OUTPUT_MAIN, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "count"])
        for term, count in mid_sentence_counter.most_common():
            writer.writerow([term, count])

    # Write sentence-start CSV sorted by frequency descending
    with open(OUTPUT_START, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "count"])
        for term, count in start_sentence_counter.most_common():
            writer.writerow([term, count])

    print("=== OUTPUT ===")
    print(f"Unique mid-sentence terms: {len(mid_sentence_counter)}")
    print(f"Unique sentence-start terms: {len(start_sentence_counter)}")


if __name__ == "__main__":
    main()