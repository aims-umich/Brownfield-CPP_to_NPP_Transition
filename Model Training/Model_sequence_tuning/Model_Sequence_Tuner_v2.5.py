import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load and preprocess the data
df_x = pd.read_csv("final_X.csv")
X = df_x.iloc[:, 1:5]  # 4 features (coordinate x-y, state, and county)
Y_layer1 = df_x.iloc[:, 5:27]  # Y data (22 features)

binary_indices = [3, 5, 9, 10, 11, 12, 13, 21]
float_indices = [i for i in range(22) if i not in binary_indices]
Y_layer1_float = Y_layer1.iloc[:, float_indices]
Y_layer1_binary = Y_layer1.iloc[:, binary_indices]

# Scaling
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_Y_float = StandardScaler()
Y_layer1_float_scaled = scaler_Y_float.fit_transform(Y_layer1_float)
Y_layer1_concat = np.concatenate([Y_layer1_float_scaled, Y_layer1_binary.values], axis=1)

X_train, X_test, Y_layer1_train, Y_layer1_test = train_test_split(X_scaled, Y_layer1_concat, test_size=0.2, random_state=42)
Y1_train, Y2_train = Y_layer1_train[:, :len(float_indices)], Y_layer1_train[:, len(float_indices):]
Y1_test, Y2_test = Y_layer1_test[:, :len(float_indices)], Y_layer1_test[:, len(float_indices):]

# Model creation function
def create_model(layer_neurons, learning_rate):
    inputs = Input(shape=(4,))
    x = Dense(layer_neurons[0], activation='relu')(inputs)
    for neurons in layer_neurons[1:]:
        x = Dense(neurons, activation='relu')(x)

    y1_output = Dense(14, activation='linear', name="y1_output")(x)
    y2_output = Dense(8, activation='sigmoid', name="y2_output")(x)
    model = Model(inputs=inputs, outputs=[y1_output, y2_output])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss={'y1_output': 'mse', 'y2_output': 'binary_crossentropy'},
                  metrics={'y1_output': 'mse', 'y2_output': 'accuracy'})
    return model

# Evaluate model function
def evaluate_model(layer_neurons, learning_rate):
    model = create_model(layer_neurons, learning_rate)
    history = model.fit(X_train, {'y1_output': Y1_train, 'y2_output': Y2_train},
                        epochs=20, batch_size=1, verbose=0)  # No validation split
    
    # Prediction and rescaling for Y1
    Y1_pred, Y2_pred = model.predict(X_test, verbose=0)
    Y1_pred_rescaled = scaler_Y_float.inverse_transform(Y1_pred)
    Y2_pred_binary = (Y2_pred >= 0.5).astype(int)
    Y1_test_rescaled = scaler_Y_float.inverse_transform(Y1_test)
    
    # Calculate R2 scores for Y1 and Y2
    r2_y1 = r2_score(Y1_test_rescaled, Y1_pred_rescaled)
    r2_y2 = r2_score(Y2_test, Y2_pred_binary)
    avg_r2 = (r2_y1*14 + r2_y2*8) / 22
    return avg_r2, layer_neurons, learning_rate

# Hyperparameter tuning routine
def hyperparameter_tuning(num_iterations=20):
    # Initialize base parameters
    base_params = {'layer_neurons': [225, 175, 150, 175, 175, 175], 'learning_rate': 0.00102487}
    results = []

    for iteration in range(num_iterations):
        base_layer_neurons = base_params['layer_neurons']
        base_learning_rate = base_params['learning_rate']
        
        # Generate test cases for this iteration, including the base model as the 0th case
        test_cases = [(base_layer_neurons, base_learning_rate)]  # Base model as 0th case

        # Generate perturbations for each layer's neurons
        for i in range(len(base_layer_neurons)):
            # Proposal to decrease neurons in layer i
            new_layer_neurons = base_layer_neurons.copy()
            new_layer_neurons[i] = max(1, new_layer_neurons[i] - 25)  # Minimum neurons set to 1
            test_cases.append((new_layer_neurons, base_learning_rate))

            # Proposal to increase neurons in layer i
            new_layer_neurons = base_layer_neurons.copy()
            new_layer_neurons[i] += 25
            test_cases.append((new_layer_neurons, base_learning_rate))

        # Generate perturbations for learning rate
        test_cases.append((base_layer_neurons, base_learning_rate * 1.1))  # Increase learning rate
        test_cases.append((base_layer_neurons, base_learning_rate / 1.1))  # Decrease learning rate

        # Add a new hidden layer with neurons equal to the last layer's size
        if len(base_layer_neurons) < 10: 
            new_layer_neurons = base_layer_neurons + [base_layer_neurons[-1]]
            test_cases.append((new_layer_neurons, base_learning_rate))

        # Remove the last hidden layer
        if len(base_layer_neurons) > 1:  # Ensure at least one layer remains
            new_layer_neurons = base_layer_neurons[:-1]
            test_cases.append((new_layer_neurons, base_learning_rate))

        # Evaluate all test cases in parallel
        results_iter = Parallel(n_jobs=24)(delayed(evaluate_model)(layers, lr) for layers, lr in test_cases)

        # Set base_r2 as the R2 score of the 0th case (base model)
        base_r2 = results_iter[0][0]

        # Separate successful proposals (those that improve upon the base model's R2)
        successful_proposals = [(r2, layers, lr) for r2, layers, lr in results_iter[1:] if r2 > base_r2]

        # Update base_params with new base configuration after all successful proposals
        base_params['layer_neurons'] = base_layer_neurons
        base_params['learning_rate'] = base_learning_rate
        print(f"Iteration {iteration + 1}: New Base Avg R2 Score = {base_r2}, Parameters: {base_params}")
        
        # Update base model parameters only if there are successful proposals
        if successful_proposals:
            # 1. Learning Rate - Look for successful proposals with a different learning rate
            for _, _, lr in successful_proposals:
                if lr != base_learning_rate:
                    base_learning_rate = lr
                    break

            # 2. Neuron Counts in Each Hidden Layer
            for _, layers, _ in successful_proposals:
                # Determine the maximum index to check, based on the shorter length
                max_index = min(len(base_layer_neurons), len(layers))
                for layer_index in range(max_index):
                    # Check for a difference in the neuron count for the current layer
                    if layers[layer_index] != base_layer_neurons[layer_index]:
                        base_layer_neurons[layer_index] = layers[layer_index]
                        break

            # 3. Layer Count - Check for adding or removing layers
            for _, layers, _ in successful_proposals:
                if len(layers) > len(base_layer_neurons):
                    # Adding a layer: replicate the neuron count of the last layer
                    base_layer_neurons.append(base_layer_neurons[-1])
                    break
                elif len(layers) < len(base_layer_neurons):
                    # Removing the last layer
                    base_layer_neurons = base_layer_neurons[:-1]
                    break

        # Write the base model parameters and R2 score to the CSV after each iteration
        with open("hyperparameter_tuning_results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, base_params['layer_neurons'], base_params['learning_rate'], base_r2])

    print("Hyperparameter tuning completed.")
    return base_params

# Run hyperparameter tuning
final_base_params = hyperparameter_tuning()
print("Final Base Parameters:", final_base_params)