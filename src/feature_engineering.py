import pandas as pd 
from sklearn.preprocessing import OneHotEncoder # type: ignore
import numpy as np
from collections import Counter

# --------------- Baseline Feature Engineering ----------------

# function to calculate the amino acid composition of a protein sequence

def amino_acid_comp(sequence):

    amino_acids = list("ARNDCQEGHILKMFPSTWYV")
    seq = sequence.upper()
    comp = {aa: 0 for aa in amino_acids}
    total = len(seq)

    for aa in seq:
        if aa in comp:
            comp[aa] += 1

    for aa in comp:
        comp[aa] /= total if total > 0 else 0
    
    return comp

# function to calculate the composition vector of a protein sequence

def comp_vector(sequence):
    comp = amino_acid_comp(sequence)
    return np.array(list(comp.values()))

# function to calculate the k-mer composition of a protein sequence    

def get_kmer_counts(sequence, k = 3):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    counts = Counter(kmers)

    total = sum(counts.values())

    return {kmer: count/total for kmer, count in counts.items()}

# function to build k-mer vocabulary of a protein sequence

def build_kmer_vocab(sequences, k = 3, min_count = 1):
    
    vocab_counter = Counter()

    for seq in sequences:
        kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        vocab_counter.update(kmers)

    vocab = {kmer for kmer, count in vocab_counter.items() if count >= min_count}

    return sorted(vocab)

def kmer_feature_vector(sequence, k = 3, vocab = None):
    if vocab is None:
        vocab = build_kmer_vocab([sequence], k = k)
    
    kmer_counts = get_kmer_counts(sequence, k = k)

    feature_vector = np.array([kmer_counts.get(kmer, 0) for kmer in vocab], dtype=float)

    l1_norm_feature_vector = feature_vector / np.linalg.norm(feature_vector, ord = 1)       # L1 normalization (better since entries sum to 1 and we can use that for probabilities)

    return l1_norm_feature_vector

def build_vocab():

    vocab = {
        "<PAD>":    0,
        "<UNK>":    1, 
        "<S>":      2,      # start token
        "</S>":     3,      # end token    
        "A":        4,
        "R":        5,
        "N":        6,
        "D":        7,
        "C":        8,
        "Q":        9,
        "E":        10,
        "G":        11,
        "H":        12,
        "I":        13,
        "L":        14,
        "K":        15,
        "M":        16,
        "F":        17,
        "P":        18,
        "S":        19,
        "T":        20,
        "W":        21,
        "Y":        22,
        "V":        23,
        "B":        24,     # ambiguous: D or N
        "Z":        25,     # ambiguous: E or Q    
        "J":        26,     # ambiguous: I or L
        "X":        27,     # any
    }

    return vocab

def tokenize_sequence(sequence, vocab):

    tokens = [vocab.get("<S>")]

    for aa in sequence:
        tokens.append(vocab.get(aa, vocab.get("<UNK>")))
    tokens.append(vocab.get("</S>"))

    return tokens

if __name__ == "__main__":

    # test the functions

    sequence = "MTEITAAMVKELRESTGAGMMDCKNALSETNGDFDKAXQLLRBZZGLGKAAKKADRLAAEG"

    print("Amino acid composition:")
    print(amino_acid_comp(sequence))
    print()

    print("Composition vector:")
    print(comp_vector(sequence))
    print()

    print("3-mer composition:")
    print(get_kmer_counts(sequence))
    print()

    print("3-mer vocabulary:")
    vocab = build_kmer_vocab([sequence], k = 3)
    print(vocab)
    print()

    print("3-mer feature vector:")
    print(kmer_feature_vector(sequence, k = 3, vocab = vocab))
    print()

    print("Tokenized sequence:")
    vocab = build_vocab()
    print(tokenize_sequence(sequence, vocab))
    print()