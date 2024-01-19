import numpy as np
import pandas as pd
import joblib
import os

def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def generate_level_configurations(num_samples=100):
    # Randomly generate level configurations
    averageWidths = np.random.uniform(3, 4, num_samples)
    averageHeights = np.random.uniform(3, 4, num_samples)
    rectNumber = np.random.randint(4, 6, num_samples)
    minPrefabDistance = np.random.uniform(5, 5, num_samples)
    numGoodPrefabs = np.random.randint(0, 10, num_samples)
    numBadPrefabs = np.random.randint(0, 10, num_samples)
    averageDistanceGood = np.random.uniform(5, 6, num_samples)
    averageDistanceBad = np.random.uniform(5, 6, num_samples)
    pathfindingObstacles = np.random.randint(0, 10, num_samples)
    numberOfEnemies = np.random.randint(0, 10, num_samples)
    minX = np.random.uniform(-10, -8, num_samples)
    maxX = np.random.uniform(6, 8, num_samples)
    minY = np.random.uniform(0, 2, num_samples)
    maxY = np.random.uniform(5, 10, num_samples)

    return np.column_stack((averageWidths, averageHeights, rectNumber, minPrefabDistance, numGoodPrefabs, numBadPrefabs, averageDistanceGood, averageDistanceBad, pathfindingObstacles, numberOfEnemies, minX, maxX, minY, maxY))

def generate_playable_level_data(model, scaler, level_configurations):
    # Feature names as per the trained model
    feature_names = ['averageWidths_Data_0', 'averageHeights_Data_0', 'RectNumber_Data_0', 
                     'minPrefabDistance_Data_0', 'numGoodPrefabs_Data_0', 'numBadPrefabs_Data_0', 
                     'averageDistanceGood_Data_0', 'averageDistanceBad_Data_0', 'PathfindingObstacles_Data_0', 
                     'numberOfEnemies_Data_0', 'MinX_Data_0', 'MaxX_Data_0', 'MinY_Data_0', 'MaxY_Data_0']


    # Create a DataFrame from the random configurations
    input_df = pd.DataFrame(level_configurations, columns=feature_names)

    # Scale the generated data using the same scaler used during training
    scaled_input = scaler.transform(input_df)

    # Predict using the model
    predictions = model.predict(scaled_input)

    # Split predictions into attempts closeness and end trigger reached predictions
    attempts_closeness_predictions = predictions[:, 0]  # First column for attempts closeness
    end_trigger_reached_predictions = predictions[:, 1]  # Second column for end trigger reached

    input_df['Attempts_Closeness'] = attempts_closeness_predictions
    input_df['EndTriggerReached'] = end_trigger_reached_predictions

    # Filter out levels based on predictions
    # Adjust the thresholds to be more lenient
    playable_levels = input_df[(input_df['Attempts_Closeness'] <= 2.0) & (input_df['EndTriggerReached'] >= 0.31)]
    return playable_levels

def save_to_csv(dataframe, filename):
    # Define the file path
    file_path = os.path.join('MaRiNo2D_Data', filename)
    # Save the dataframe to a CSV file with semicolon separators
    dataframe.to_csv(file_path, sep=';', index=False)
    print(f"Data saved to {file_path}")


if __name__ == "__main__":
    model_path = 'trained_model.pkl'  # Path to the trained model
    scaler_path = 'scaler.pkl'  # Path to the saved scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    num_samples = 10000  # Number of level configurations to generate and evaluate

    random_level_configurations = generate_level_configurations(num_samples)
    playable_levels = generate_playable_level_data(model, scaler, random_level_configurations)

    print("Generated Playable Level Data:\n", playable_levels)
    
    # Save the playable levels data to a CSV file
    save_to_csv(playable_levels, 'playable_levels.csv')
