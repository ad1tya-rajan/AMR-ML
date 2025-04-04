import glob
import os
import pandas as pd         # type: ignore
from Bio import SeqIO       # type: ignore

def parse_fasta(fasta_file, output_csv = None):

    records = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        header_parts = record.description.split("|")
        if len(header_parts) >= 10:
            drug_class = header_parts[9].strip()
            gene_name = header_parts[4].strip()
        else:
            drug_class = "Unknown"
            gene_name = "Unknown"

        records.append({
            "sequence": str(record.seq),
            "drug_class": drug_class,
            "gene_name": gene_name
        })
    
    df = pd.DataFrame(records)

    if output_csv:
        processed_dir = "/home/cvm-alamlab/Desktop/Aditya/AMR_Project/AMR-ML/data/processed"
        os.makedirs(processed_dir, exist_ok=True)           # create the directory if it doesn't exist
        output_csv_path = os.path.join(processed_dir, output_csv)
        df.to_csv(output_csv_path, index=False)
        print(f"Data saved to {output_csv_path}")
    
    return df                                               # return a pandas dataframe

if __name__ == "__main__":
    
    fasta_file = "/home/cvm-alamlab/Desktop/Aditya/AMR_Project/AMR-ML/data/raw/AMRProt.fa"      # or test.fa
    output_csv = "test.csv"
    df = parse_fasta(fasta_file, output_csv=output_csv)
    
    print(df.head())
    print("DF shape: ", df.shape)