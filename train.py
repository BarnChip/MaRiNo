import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Function to read and preprocess data from a text file
def read_and_preprocess_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Each line contains multiple values separated by spaces, which are now converted to floats
        data_point = list(map(float, line.split()))

        # Add the data point to the list
        data.append(data_point)

    return data


# List of file paths for different categories
file_paths = [
    'MaRiNo2D_Data\EndTriggerReached.txt',
    r'MaRiNo2D_Data\averageWidths.txt',
    r'MaRiNo2D_Data\averageHeights.txt',
    'MaRiNo2D_Data\RectNumber.txt',
    
    
    'MaRiNo2D_Data\minPrefabDistance.txt',
    r'MaRiNo2D_Data\numGoodPrefabs.txt',
    r'MaRiNo2D_Data\numBadPrefabs.txt',
    r'MaRiNo2D_Data\averageDistanceGood.txt',
    
    
    r'MaRiNo2D_Data\averageDistanceBad.txt',
    'MaRiNo2D_Data\PathfindingObstacles.txt',
    r'MaRiNo2D_Data\numberOfEnemies.txt',
    'MaRiNo2D_Data\MinX.txt',
    
    
    'MaRiNo2D_Data\MaxX.txt',
    'MaRiNo2D_Data\MinY.txt',
    'MaRiNo2D_Data\MaxY.txt',
    r'MaRiNo2D_Data\Attempts.txt'

]

# Initialize a list to store the dataframes
dfs = []

# Read and preprocess data from each file and create a dataframe
for file_path in file_paths:
    data = read_and_preprocess_data(file_path)
    label = file_path.split('\\')[-1].split('.')[0]
    df = pd.DataFrame(data, columns=[f'{label}_Data_{i}' for i in range(len(data[0]))])
    dfs.append(df)

# Aligning all dataframes to the same length
max_length = max(len(df) for df in dfs)
for i, df in enumerate(dfs):
    if len(df) < max_length:
        dfs[i] = df.reindex(range(max_length), fill_value=0)

# Concatenate dataframes horizontally
final_df = pd.concat(dfs, axis=1)
final_df['Attempts_Closeness'] = final_df['Attempts_Data_0'].apply(lambda x: abs(x - 3))

# Two target variables
y = final_df[['Attempts_Closeness', 'EndTriggerReached_Data_0']]
X = final_df.drop(columns=['Attempts_Data_0', 'Attempts_Closeness', 'EndTriggerReached_Data_0'])

# Specify the file path for the CSV
csv_file_path = 'final_dataframe.csv'

# Write the DataFrame to a CSV file
final_df.to_csv(csv_file_path, index=False)

print(f"DataFrame saved as CSV at: {csv_file_path}")


print(y)
print(X)
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Calculate and print the mean absolute error for each target variable
mae_attempts = mean_absolute_error(y_test['Attempts_Closeness'], y_pred[:, 0])
mae_end_trigger = mean_absolute_error(y_test['EndTriggerReached_Data_0'], y_pred[:, 1])

print(f"Mean Absolute Error - Attempts: {mae_attempts}")
print(f"Mean Absolute Error - End Trigger Reached: {mae_end_trigger}")

model_path = 'trained_model.pkl'
scaler_path = 'scaler.pkl'
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("Random Forest Regression for multiple outputs completed and model saved.")