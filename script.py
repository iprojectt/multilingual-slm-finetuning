
# ------------------------------ ------------------------------ ------------------------------
# This scripts Outputs cleaned text files for Hindi (Sangraha) and English (mC4) datasets.

# Configured datasets with their Hugging Face IDs, subsets, splits, and target token counts.
# I am streaming datasets to avoid memory issues and processesing them line by line.
# Logging progress every 10 million tokens and stopping once the target token count is reached.

# I am using a simple space-separated token count to know how much data to download.
#  This gives an easy way to track progress and make sure we reach the target number 
# of tokens without doing any complicated calculations, even for large datasets.
# ------------------------------ ------------------------------ ------------------------------

from datasets import load_dataset
import os
# -------- CONFIG (Hub IDs) --------
DATASETS = {
    # "mC4_hi": ("allenai/c4", "hi", "train", 600_000_000, "hindi.txt"),
    # "OSCAR_hi": ("oscar-corpus/OSCAR-2301", "hi", "train", 300_000_000, "hindi.txt"),
    # "CC100_hi": ("yhavinga/cc100", "hi", "train", 610_000_000, "hindi.txt"),
    "Sangraha_hi": ("ai4bharat/sangraha", "verified", "hin", 1_300_000_000, "hindi_sangraha.txt"),
    "mC4_en": ("allenai/c4", "en", "train", 1_550_000_000, "english.txt"),
}
PROGRESS_STEP = 10_000_000  # log every 10M tokens
for name, (hf_id, subset, split, target_tokens, out_file) in DATASETS.items():
    print(f"\n Processing {name} → Target: {target_tokens:,} tokens → {out_file}")
    ds = load_dataset(hf_id, subset, split=split, streaming=True)
    token_count = 0
    last_logged = 0
    mode = "a" if os.path.exists(out_file) else "w"
    with open(out_file, mode, encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "").replace("\n", " ").strip()
            if not text:
                continue
            n_tokens = len(text.split())
            token_count += n_tokens
            f.write(text + "\n")
            if token_count - last_logged >= PROGRESS_STEP:
                print(f"   → {token_count:,} tokens so far for {name}...")
                last_logged = token_count
            if token_count >= target_tokens:
                break
    print(f" Finished {name}: wrote ~{token_count:,} tokens to {out_file}")
print("\n All datasets complete!")
print(" → hindi_sangraha.txt (~1.3B tokens from Sangraha Hindi Verified)")
print(" → english.txt (~1.5B tokens from mC4 only)")


# ------------------------------ ------------------------------ ------------------------------
# DOWNLOADED AWADHI DATASET FROM MANY RESOURCES (MENTIONED IN REPORT)
# AND THEN COMBINED ALL INTO A AWADHI_LARGE.txt FILE
# ------------------------------ ------------------------------ ------------------------------


# From terminal install: 
# 1. pip install sentencepiece
# 2. pip install indicnlp
# 3. pip install indic-nlp-library


# ------------------------------ ------------------------------ ------------------------------
# These tools are used to ensure accurate and language-specific sentence
#  segmentation for both English and Indic languages, which is critical 
# for preparing clean and structured text for language model training.
# ------------------------------ ------------------------------ ------------------------------

import unicodedata
import re

def clean_and_normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFC", line)
    line = re.sub(r'[\u200B-\u200D\uFEFF]', '', line)  # zero-width chars
    line = ''.join(ch for ch in line if ch.isprintable() or ch in ['\n', '\t'])
    return line.strip()

def split_english_sentences(text: str):
    # Split on . ? ! followed by space or line end
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def process_file(input_file: str, output_file: str, chunk_size: int = 10000):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        buffer = []
        prev_blank = False
        for line in fin:
            line = clean_and_normalize_line(line)
            if line:
                buffer.append(line)
            else:
                if not prev_blank:
                    fout.write("\n")
                prev_blank = True

            if len(buffer) >= chunk_size:
                text = " ".join(buffer)
                sentences = split_english_sentences(text)
                for sent in sentences:
                    fout.write(sent + "\n")
                buffer = []
                prev_blank = False

        if buffer:
            text = " ".join(buffer)
            sentences = split_english_sentences(text)
            for sent in sentences:
                fout.write(sent + "\n")

    print(f"Processed English saved: {output_file}")

if __name__ == "__main__":
    process_file("english.txt", "english_cleaned.txt")





import unicodedata
import re
from indicnlp.tokenize import sentence_tokenize

def clean_and_normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFC", line)
    line = re.sub(r'[\u200B-\u200D\uFEFF]', '', line)  # remove zero-width chars
    line = ''.join(ch for ch in line if ch.isprintable() or ch in ['\n', '\t'])
    return line.strip()

def process_file(input_file: str, output_file: str, language: str = "hi", chunk_size: int = 10000):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        buffer = []
        for line in fin:
            line = clean_and_normalize_line(line)
            if line:
                buffer.append(line)

            # process in chunks
            if len(buffer) >= chunk_size:
                text = " ".join(buffer)
                sentences = sentence_tokenize.sentence_split(text, lang=language)
                for sent in sentences:
                    fout.write(sent.strip() + "\n")
                buffer = []

        # process remaining lines
        if buffer:
            text = " ".join(buffer)
            sentences = sentence_tokenize.sentence_split(text, lang=language)
            for sent in sentences:
                fout.write(sent.strip() + "\n")

    print(f"Processed {language.upper()} (streaming): {output_file}")

if __name__ == "__main__":
    process_file("hindi_sangraha.txt", "hindi_cleaned.txt", language="hi")
    # For Awadhi (reusing Hindi rules)
    process_file("awadhi_large.txt", "awadhi_cleaned.txt", language="hi")




# ------------------------------ ------------------------------ ------------------------------
# Now we are calculating total sentences in each dataset
# ------------------------------ ------------------------------ ------------------------------

files = ["awadhi_cleaned.txt", "english_cleaned.txt", "hindi_cleaned.txt"]

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    print(f"{file_path} has {line_count} lines.")



# ------------------------------ ------------------------------ ------------------------------
# Now merging the subset of text files into a single temporary file:
#  combined_corpus_temp.txt and after that Shuffling the merging file and 
# saving as spm_corpus.txt
# ------------------------------ ------------------------------ ------------------------------

import random
import os
def reservoir_sample(input_file, num_samples, output_file):
    """
    Memory-efficient random sampling using reservoir sampling.
    """
    reservoir = []
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i < num_samples:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < num_samples:
                    reservoir[j] = line
    with open(output_file, "w", encoding="utf-8") as fout:
        for line in reservoir:
            fout.write(line + "\n")

def memory_efficient_merge_and_shuffle(files, num_final_samples, output_file, temp_file_path):
    """
    Merges files and then shuffles them using a memory-efficient reservoir sampling approach.
    This avoids loading all lines into memory at once.
    """
    # Step 1: Merge all partial files into a single temporary file
    print(f"Merging files into a single temporary file: {temp_file_path}...")
    with open(temp_file_path, "w", encoding="utf-8") as merged_file:
        for fname in files:
            with open(fname, "r", encoding="utf-8") as current_file:
                for line in current_file:
                    merged_file.write(line)
    # Step 2: Use reservoir sampling to shuffle the merged file
    # and select a fixed number of samples for the final output.
    # This acts as the "shuffle" by producing a random, final subset.
    print(f"Shuffling the merged file and saving to {output_file}...")
    reservoir = []
    with open(temp_file_path, "r", encoding="utf-8") as merged_file:
        for i, line in enumerate(merged_file):
            line = line.strip()
            if not line:
                continue
            if i < num_final_samples:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < num_final_samples:
                    reservoir[j] = line
    random.shuffle(reservoir) # Final in-memory shuffle of the small reservoir
    with open(output_file, "w", encoding="utf-8") as fout:
        for line in reservoir:
            fout.write(line + "\n")
    # Clean up temporary files
    os.remove(temp_file_path)

if __name__ == "__main__":
    random.seed(42)
    # Step 1: sample each language
    # Note: These are now intermediate files, not the final output.
    reservoir_sample("english_cleaned.txt", 5000000, "english_part.txt")
    reservoir_sample("hindi_cleaned.txt", 3500000, "hindi_part.txt")
    reservoir_sample("awadhi_cleaned.txt", 1500000, "awadhi_part.txt")
    # Step 2: merge & shuffle
    # We'll use a new function that is truly memory-efficient.
    # We'll set the total final sample size to 10,000,000. (10 million sentences)
    total_samples = 5000000 + 3500000 + 1500000
    temp_file = "combined_corpus_temp.txt"
    memory_efficient_merge_and_shuffle(
        ["english_part.txt", "hindi_part.txt", "awadhi_part.txt"],
        total_samples,
        "spm_corpus.txt",
        temp_file
    )
    # Clean up intermediate files
    os.remove("english_part.txt")
    os.remove("hindi_part.txt")
    os.remove("awadhi_part.txt")
    print("Merged and shuffled corpus saved as spm_corpus.txt")


#  ------------------------------ ------------------------------ ------------------------------
#  Now training the SentencePiece tokenizer on our sampled & shuffled data
#  ------------------------------ ------------------------------ ------------------------------

import sentencepiece as spm
spm_train_params = (
    "--input=spm_corpus.txt "
    "--model_prefix=spm_unigram_50k "
    "--vocab_size=50000 "
    "--model_type=unigram "
    "--character_coverage=0.9995 "
    "--byte_fallback=true "
    "--input_sentence_size=5000000 "
    "--shuffle_input_sentence=true"
)
# Runs the SentencePiece training command
spm.SentencePieceTrainer.train(spm_train_params)



# ------------------------------ ------------------------------ ------------------------------
# Now we will run our tokenizer on some random sample sentences from all dataset
# ------------------------------ ------------------------------ ------------------------------

import random
sp = spm.SentencePieceProcessor(model_file='spm_unigram_50k.model')
# Taking random sample lines from a file
def sample_lines(file_path, num_samples=5):
    # Count total lines first
    with open(file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    sample_indices = set(random.sample(range(total_lines), min(num_samples, total_lines)))
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in sample_indices:
                line = line.strip()
                if line:
                    samples.append(line)
            if len(samples) >= num_samples:
                break
    return samples

# ------------------------------ ------------------------------ ------------------------------
# Function to tokenize and the print the sample sentences and tokenized output 
# ------------------------------ ------------------------------ ------------------------------

def show_tokenization(language, file_path, num_samples=5):
    lines = sample_lines(file_path, num_samples)
    print(f"\n--- {language} Tokenization (from dataset) ---")
    for line in lines:
        tokens = sp.encode(line, out_type=str)
        print(f"Sentence: {line}")
        print(f"Tokens: {tokens}\n")


english_file = 'english_cleaned.txt'
hindi_file = 'hindi_cleaned.txt'
awadhi_file = 'awadhi_cleaned.txt'
# Show tokenization samples
show_tokenization("English", english_file, num_samples=5)
show_tokenization("Hindi", hindi_file, num_samples=5)
show_tokenization("Awadhi", awadhi_file, num_samples=5)






# ------------------------------ ------------------------------ ------------------------------
# Function to evluate token distribution and other metrices
# ------------------------------ ------------------------------ ------------------------------

import regex as re
sp = spm.SentencePieceProcessor(model_file='spm_unigram_50k.model')
# Helper functions
def is_deva(ch):
    """Check if character is Devanagari"""
    return "\u0900" <= ch <= "\u097F"

def evaluate_file(file_path, language_name, sp, limit=None):
    total_chars = 0
    deva_chars = 0
    total_tokens = 0
    unk_tokens = 0
    byte_fallback_tokens = 0
    total_words = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue

            total_chars += len(line)
            deva_chars += sum(is_deva(c) for c in line)
            words = [w for w in re.split(r'\s+', line) if w]
            total_words += len(words)

            tokens = sp.encode(line, out_type=str)
            total_tokens += len(tokens)
            unk_tokens += tokens.count('<unk>')
            byte_fallback_tokens += sum(t.startswith('<0x') or t.startswith('▁<0x') for t in tokens)

    # Metrics
    tokens_per_word = total_tokens / max(1, total_words)
    deva_fraction = deva_chars / max(1, total_chars)
    byte_fallback_fraction = byte_fallback_tokens / max(1, total_tokens)
    unk_fraction = unk_tokens / max(1, total_tokens)

    print(f"\n=== {language_name} ===")
    print(f"Total chars: {total_chars}")
    print(f"Devanagari fraction: {deva_fraction:.3f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens per word: {tokens_per_word:.3f}")
    print(f"<unk> tokens: {unk_tokens} ({unk_fraction:.4f})")
    print(f"Byte-fallback tokens: {byte_fallback_tokens} ({byte_fallback_fraction:.4f})")

    return total_tokens


english_file = 'english_cleaned.txt'
hindi_file = 'hindi_cleaned.txt'
awadhi_file = 'awadhi_cleaned.txt'
# Evaluate each language
tokens_eng = evaluate_file(english_file, 'English', sp)
tokens_hin = evaluate_file(hindi_file, 'Hindi', sp)
tokens_awa = evaluate_file(awadhi_file, 'Awadhi', sp)

total_tokens = tokens_eng + tokens_hin + tokens_awa

print("\n=== Overall Token Distribution ===")
print(f"English tokens: {tokens_eng} ({tokens_eng/total_tokens:.2%})")
print(f"Hindi tokens: {tokens_hin} ({tokens_hin/total_tokens:.2%})")
print(f"Awadhi tokens: {tokens_awa} ({tokens_awa/total_tokens:.2%})")
print(f"Total tokens: {total_tokens}")


