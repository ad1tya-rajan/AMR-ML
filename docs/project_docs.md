**Documentation:**

- General Pipeline: 
1. Data cleaning and preprocessing: We will use .tsv files generated after running AMRFinder (ncbi-amrfinderplus) and Abricate on 1,996 FASTA files (assembled genome samples) curated by Dr. Alam. Cleaning involves renaming columns for easier encoding and removing columns that do not contain critical information.

2. Feature extraction: Once the data has been cleaned and processed, we will extract any meaningful features from the data. This includes numerical (percentage identity, coverage, etc.) and categorical features (antibiotic class).

3. Model development and training: We will train a baseline model (probably XGBoost or Random Forest), and develop and train a custom transformer-based model. The transformer architecture still needs to be worked out. Training will involve traditional data splitting methods (like sklearn).

4. Model evaluation: We will evaluate the models with various metrics like precision, recall, accuracy, ROC-AUC score, etc.

5. Deployment: Based on the completion and performance of the model, we may deploy this project as a web app or CLI tool.

- Transformer Architecture:

We plan on using a transformer framework, particularly a Biderctional Encoder Representations from Transformers (BERT). Additionally, we may use a more recent framework built on top of BERT - A Robustly Optimized BERT Training Approach (RoBERTa). Existing frameworks employ a transformer with a Bidirectional Gated Recurrent Unit (BiGRU). 
