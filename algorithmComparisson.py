import os
import pandas as pd
import itertools
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Create output folder
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Load data from CSV file
data = pd.read_csv("",
                   header=None)

# Drop missing values
data = data.dropna()

# Split input features (X) and output labels (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(output_folder, 'minmax_scaler.pkl'))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Expanded hyperparameter grid for each model
param_sets = {
    'SVM': itertools.product([0.1, 1.0, 10.0], ['linear', 'rbf', 'poly'], [2, 3, 4]),
    'Random Forest': itertools.product([50, 100, 200], [None, 5, 10], [2, 5, 10]),
    'XGBoost': itertools.product([50, 100, 200], [2, 3, 5], [0.05, 0.1, 0.2]),
    'Logistic Regression': itertools.product([0.01, 1.0, 10.0], ['lbfgs', 'liblinear', 'saga']),
    'k-NN': itertools.product([3, 5, 10], ['uniform', 'distance']),
    'Naive Bayes': [{}]
}

# Store results
results = []

# Train and evaluate models
for name, param_combinations in param_sets.items():
    for i, params in enumerate(param_combinations):
        print(f"Training {name} with parameter set {i + 1}")

        if name == 'SVM':
            model = SVC(C=params[0], kernel=params[1], degree=params[2] if params[1] == 'poly' else 3, probability=True,
                        random_state=42)
        elif name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=params[0], max_depth=params[1], min_samples_split=params[2],
                                           random_state=42)
        elif name == 'XGBoost':
            model = XGBClassifier(n_estimators=params[0], max_depth=params[1], learning_rate=params[2],
                                  use_label_encoder=False, eval_metric='logloss', random_state=42)
        elif name == 'Logistic Regression':
            model = LogisticRegression(C=params[0], solver=params[1], random_state=42)
        elif name == 'k-NN':
            model = KNeighborsClassifier(n_neighbors=params[0], weights=params[1])
        elif name == 'Naive Bayes':
            model = GaussianNB()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Save model
        model_path = os.path.join(output_folder, f"{name.lower().replace(' ', '_')}_set{i + 1}.pkl")
        joblib.dump(model, model_path)

        # Store results
        results.append({
            'Model': f'{name} Set {i + 1}',
            'Base Model': name,
            'Accuracy': accuracy,
            'Classification Report': classification_report(y_test, predictions, output_dict=True, zero_division=1),
            'Confusion Matrix': confusion_matrix(y_test, predictions)
        })

# Neural Network Hyperparameter Grid
nn_params = list(itertools.product(
    [8, 16, 32, 64],  # Layer 1 Neurons
    [0.3, 0.5, 0.6],  # Dropout Rate 1
    [8, 16, 32, 64],  # Layer 2 Neurons
    [0.3, 0.4, 0.5]  # Dropout Rate 2
))

# Train Neural Networks
for i, params in enumerate(nn_params):
    print(f"Training Neural Network with parameter set {i + 1}/{len(nn_params)}: {params}")
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(params[0], activation='relu'),
        Dropout(params[1]),
        Dense(params[2], activation='relu'),
        Dropout(params[3]),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=500, batch_size=16, verbose=0, validation_data=(X_test, y_test))
    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

    # Save model
    model.save(os.path.join(output_folder, f'neural_network_model_set{i + 1}.h5'))

    # Store results
    results.append({
        'Model': f'Neural Network Set {i + 1}',
        'Base Model': 'Neural Network',
        'Accuracy': accuracy
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(output_folder, "model_comparison_results.csv"), index=False)

# Plot accuracy comparison grouped by model type
plt.figure(figsize=(12, 7), dpi=200)
avg_accuracy = df_results.groupby("Base Model")["Accuracy"].mean().reset_index()
sns.barplot(x='Accuracy', y='Base Model', data=avg_accuracy.sort_values(by='Accuracy', ascending=False),
            edgecolor='black', linewidth=1.5, palette='viridis')
plt.xlabel("Average Accuracy", fontsize=14, fontweight='bold')
plt.ylabel("Model Type", fontsize=14, fontweight='bold')
plt.title("Average Accuracy by Model Type", fontsize=16, fontweight='bold')
plt.savefig(os.path.join(output_folder, "accuracy_comparison_avg.png"))
plt.show()

print("All results and models saved in the 'output' folder.")
