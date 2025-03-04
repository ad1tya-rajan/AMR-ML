import glob
import os
import pandas as pd

def load_tsv_files(dir):
    tsv_files = glob.glob(os.path.join(dir, "*.tsv"))
    dfs = []

    for file in tsv_files:
        try:
            df = pd.read_csv(file, sep = "\t")
            sample_id = os.path.splitext(os.path.basename(file))[0]
            df["sample_id"] = sample_id

        except Exception as e:
            print(f"Error processing {file}: ", e)

        if dfs:
            master_df = pd.concat(dfs, ignore_index=True)
            return master_df
        else:
            raise ValueError("No TSV files were found.")
        
def clean_data(df):
    # renaming columnsfor easier encoding
    # drop cols with missing critical data

    return df

if __name__ == "__main__":
    dir = "database path"       # <--- change to tsv dir
    master_df = load_tsv_files(dir)
    master_df = clean_data(master_df)
    print("Master dataframe head: \n", master_df.head())