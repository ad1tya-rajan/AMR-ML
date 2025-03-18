import glob
import os
import pandas as pd         # type: ignore
from Bio import SeqIO       # type: ignore

def parse_fasta(file, output_csv = None):

    records = []

    for record in SeqIO.parse(file, "fasta"):

        # extract data from the header
        header = record.description.lstrip(">")
        fields = header.split("|")

        if len(fields) >= 4:
            accession = fields[1].strip()           # accession number
            aro_id = fields[2].strip()              # ARO ID
            gene_field = fields[3].strip()          # gene name and description

            gene_name = gene_field.split()[0]
            description = " ".join(gene_field.split()[1:]).strip() if len(gene_field.split()) > 1 else ""
        
        else:
            accession = ""
            aro_id = ""
            gene_name = ""
            description = header        # fallback to using the entire header as the gene name

        records.append({
            "accession": accession,
            "aro_id": aro_id,
            "gene_name": gene_name,
            "description": description,
            "sequence": str(record.seq)
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
    
    fasta_file = "/home/cvm-alamlab/Desktop/Aditya/AMR_Project/AMR-ML/data/raw/test.fasta"
    output_csv = "test.csv"
    df = parse_fasta(fasta_file, output_csv=output_csv)
    
    print(df.head())
    print("DF shape: ", df.shape)