import tensorflow as tf
import numpy as np
import os
from data_preparation import load_and_preprocess_data

def generate_peptides(model_path, latent_dim, num_peptides, char_to_index, amino_acids, max_sequence_length):
    """Generates novel peptide sequences using the trained VAE.

    Args:
        model_path (str): Path to the saved VAE model.
        latent_dim (int): Dimension of the latent space.
        num_peptides (int): Number of peptides to generate.
        char_to_index (dict): Character to index mapping.
        amino_acids (str): String of amino acids.
        max_sequence_length (int): Maximum sequence length used for padding.

    Returns:
        list: A list of generated peptide sequences.
    """

    # 1. Load the Trained Models
    vae = tf.keras.models.load_model(model_path, custom_objects={'Sampling': Sampling})

    # 2. Reverse the Character to Index Mapping
    index_to_char = {index: char for char, index in char_to_index.items()}

    # 3. Generation Loop
    generated_peptides = []
    for _ in range(num_peptides):
        # a. Sample a random vector from the latent space
        random_latent_vector = np.random.normal(0, 1, size=(1, latent_dim))

        # b. Decode the latent vector to generate a sequence
        reconstructed_sequence = vae.decode(random_latent_vector)

        # c. Convert the predicted sequence (probabilities) to amino acid indices
        predicted_indices = np.argmax(reconstructed_sequence, axis=-1)[0]  # Take the argmax along the last axis

        # d. Convert the indices to amino acids
        predicted_peptide = ''.join([index_to_char.get(idx, '') for idx in predicted_indices if idx != 0 and idx < len(amino_acids)])  # Decode and exclude padding (index 0)

        generated_peptides.append(predicted_peptide)

    return generated_peptides


if __name__ == "__main__":
    # Example Usage:
    from vae_model import Sampling
    latent_dim = 32 #MUST MATCH THE DIMENSION USED FOR TRAINING
    num_peptides = 50
    model_path = "output/models/vae.keras"  # Replace with the actual path to your saved encoder
    data_path = "data/peptides.csv"  # Replace with your data path

    # Load necessary data
    encoded_data, encoded_data_test, char_to_index, amino_acids, max_sequence_length = load_and_preprocess_data(data_path, sequence_column = "SEQUENCE")

    # Generate peptides
    generated_peptides = generate_peptides(model_path, latent_dim, num_peptides, char_to_index, amino_acids, max_sequence_length)

    # Print the generated peptides
    print("Generated Peptides:")
    for peptide in generated_peptides:
        print(peptide)
