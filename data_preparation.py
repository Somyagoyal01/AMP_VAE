import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_path, sequence_column='SEQUENCE', test_size=0.2):
    """Loads, cleans, and preprocesses the peptide data from DBAASP.

    Args:
        data_path (str): Path to the CSV file containing peptide data.
        sequence_column (str): Name of the column containing the peptide sequences.
        test_size (float): Proportion of data to use for testing

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: A numpy array of encoded peptide sequences.
            - dict: A dictionary mapping amino acids to indices.
            - str: A string containing all unique amino acids.
            - int: The maximum sequence length.
    """
    try:
        df = pd.read_csv(data_path)  # Load the CSV file
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading data: {e}")
        return None, None, None, None

    # 1. Basic Cleaning (Handle Missing Values, Remove Duplicates)
    df = df.dropna(subset=[sequence_column])  # Remove rows with missing sequences
    df = df.drop_duplicates(subset=[sequence_column])  # Remove duplicate sequences

    # 2. Sequence Length Filtering (Optional - might help VAE training)
    min_length = 5  # Adjust as needed
    max_length = 50  # Adjust as needed
    df = df[(df[sequence_column].str.len() >= min_length) & (df[sequence_column].str.len() <= max_length)]

    # 3. Character Encoding (Convert Amino Acids to Numerical Representation)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # Standard 20 amino acids
    char_to_index = {char: index for index, char in enumerate(amino_acids)}

    def encode_sequence(sequence, max_len):
        encoded = np.zeros(max_len, dtype=int) # initialize with zeros
        for i, char in enumerate(sequence):
            if char in char_to_index:
                encoded[i] = char_to_index[char]
            else:
                encoded[i] = len(amino_acids) # Use the last position for unknown amino acids
        return encoded

    # Find the maximum sequence length for padding
    max_sequence_length = df[sequence_column].str.len().max()
    # Ensure the maximum length is not greater than your selected max_length
    max_sequence_length = min(max_sequence_length, max_length)


    df['encoded_sequence'] = df[sequence_column].apply(lambda seq: encode_sequence(seq, max_sequence_length))

    # 4. Convert to Numpy Array for VAE training
    encoded_sequences = np.array(df['encoded_sequence'].tolist())

    X_train, X_test = train_test_split(encoded_sequences, test_size=test_size, random_state=42) #Splitting data
    return X_train, X_test, char_to_index, amino_acids, max_sequence_length #Returning splits as well


if __name__ == "__main__":
    # Example Usage:
    data_file = "data/peptides.csv"  # Replace with your actual file path
    X_train, X_test, char_to_index, amino_acids, max_sequence_length = load_and_preprocess_data(data_file, sequence_column='SEQUENCE')

    if X_train is not None:
        print("Shape of encoded training data:", X_train.shape)
        print("Example encoded sequence from training:", X_train[0])

        print("Shape of encoded testing data:", X_test.shape)
        print("Example encoded sequence from testing:", X_test[0])

        print("Character to index mapping:", char_to_index)
        print("Amino acids:", amino_acids)
        print("Max sequence length:", max_sequence_length)
    else:
        print("Data loading and preprocessing failed.")