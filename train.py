import tensorflow as tf
import numpy as np
from vae_model import VAE
from data_preparation import load_and_preprocess_data
import os #for creating directories

# Hyperparameters (Adjust these!)
latent_dim = 32
batch_size = 32
epochs = 10
learning_rate = 0.001
test_size = 0.2 #Size of test set,

# Data Path
data_path = "data/peptides.csv"  # Replace with your data path

# Create directories for saving models and results
output_dir = "output"
model_dir = os.path.join(output_dir, "models")
os.makedirs(model_dir, exist_ok=True)

def train_vae(latent_dim, batch_size, epochs, learning_rate, data_path, test_size):
    """Trains the VAE model.

    Args:
        latent_dim (int): Dimension of the latent space.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        data_path (str): Path to the preprocessed data file.
    """
    # 1. Load and Preprocess Data
    X_train, X_test, char_to_index, amino_acids, max_sequence_length = load_and_preprocess_data(data_path, sequence_column='SEQUENCE', test_size = test_size)
    if X_train is None:
        print("Error: Data loading failed.  Check your data path and format.")
        return None, None, None, None

    # 2.  Determine Input Dimension
    input_dim = len(char_to_index)  # Number of unique amino acids in your data

    # 3. Create VAE Model
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, amino_acids_count = len(amino_acids), max_sequence_length=max_sequence_length) #added amino_acids_count and max_sequence_length

    # 4. Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # 5. Loss Function (Reconstruction Loss)
    def reconstruction_loss(y_true, y_pred):
        """Calculates the reconstruction loss."""
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(loss)

    # 6. Training Loop
    @tf.function
    def train_step(x):
        """Performs a single training step."""
        with tf.GradientTape() as tape:
            reconstructed = vae(x)
            loss = reconstruction_loss(x, reconstructed) #reconstruction loss using x and reconstructed sequence
            loss += sum(vae.losses)  # Add KL divergence loss

        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        return loss

    # Prepare dataset
    dataset = tf.data.Dataset.from_tensor_slices(X_train) #Training dataset
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("Starting Training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, x_batch_train in enumerate(dataset):
            loss = train_step(x_batch_train)
            epoch_loss += loss.numpy()
            if step % 100 == 0: # Print loss every 100 steps
                print(f"Epoch {epoch+1}/{epochs}, Step {step}, Loss: {loss.numpy():.4f}")

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss / len(dataset):.4f}")

    # 7. Save the Trained Model
    vae.save(os.path.join(model_dir, "vae.keras")) #Saving as vae instead of encoder/decoder
    print(f"Models saved to {model_dir}")
    return vae, char_to_index, amino_acids, max_sequence_length

if __name__ == "__main__":
    trained_vae, char_to_index, amino_acids, max_sequence_length = train_vae(latent_dim, batch_size, epochs, learning_rate, data_path, test_size)
    print("Training Complete!")

    if trained_vae:
      trained_vae.summary()