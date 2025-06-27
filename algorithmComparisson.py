import os
import pandas as pd
import itertools
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Output directory
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Load and prepare data
data = pd.read_csv("AI_data_air_printing_updated_clear.csv", header=None).dropna()
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Scalers to test
scalers = {
    'minmax': MinMaxScaler(),
    'zscore': StandardScaler()
}

# Hyperparameter grids (Naive Bayes removed)
param_sets = {
    'SVM': itertools.product([0.1, 1.0, 10.0], ['linear', 'rbf', 'poly'], [2, 3, 4]),
    'Random Forest': itertools.product([50, 100, 200], [None, 5, 10], [2, 5, 10]),
    'XGBoost': itertools.product([50, 100, 200], [2, 3, 5], [0.05, 0.1, 0.2]),
    'Logistic Regression': itertools.product([0.01, 1.0, 10.0], ['lbfgs', 'liblinear', 'saga']),
    'k-NN': itertools.product([3, 5, 10], ['uniform', 'distance'])
}

# Neural network parameter grid
nn_params = list(itertools.product(
    [8, 16, 32, 64],  # layer1 neurons
    [0.3, 0.5, 0.6],  # dropout1
    [8, 16, 32, 64],  # layer2 neurons
    [0.3, 0.4, 0.5]  # dropout2
))

results = []

for scaler_name, scaler in scalers.items():
    print(f"\n=== Using scaler: {scaler_name} ===")
    # Fit and save scaler
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(output_folder, f"{scaler_name}_scaler.pkl"))

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train classical ML models
    for model_name, grid in param_sets.items():
        for idx, params in enumerate(grid, start=1):
            print(f"[{scaler_name}] Processing {model_name} set {idx}: {params}")
            if model_name == 'SVM':
                model = SVC(
                    C=params[0], kernel=params[1],
                    degree=(params[2] if params[1] == 'poly' else 3),
                    probability=True, random_state=42
                )
            elif model_name == 'Random Forest':
                model = RandomForestClassifier(
                    n_estimators=params[0],
                    max_depth=params[1],
                    min_samples_split=params[2],
                    random_state=42
                )
            elif model_name == 'XGBoost':
                model = XGBClassifier(
                    n_estimators=params[0],
                    max_depth=params[1],
                    learning_rate=params[2],
                    eval_metric='logloss',
                    random_state=42
                )
            elif model_name == 'Logistic Regression':
                model = LogisticRegression(
                    C=params[0], solver=params[1], random_state=42
                )
            else:  # k-NN
                model = KNeighborsClassifier(
                    n_neighbors=params[0], weights=params[1]
                )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            # Save model
            filename = f"{scaler_name}_{model_name.lower().replace(' ', '_')}_set{idx}.pkl"
            joblib.dump(model, os.path.join(output_folder, filename))

            # Store results
            results.append({
                'Scaler': scaler_name,
                'Base Model': model_name,
                'Set': idx,
                'Accuracy': acc,
                'Report': classification_report(y_test, preds, output_dict=True, zero_division=1),
                'Confusion Matrix': confusion_matrix(y_test, preds)
            })

    # Train neural networks
    for idx, (n1, d1, n2, d2) in enumerate(nn_params, start=1):
        print(f"[{scaler_name}] Processing Neural Network set {idx}/{len(nn_params)}: "
              f"layers=({n1},{n2}), dropouts=({d1},{d2})")
        nn = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(n1, activation='relu'), Dropout(d1),
            Dense(n2, activation='relu'), Dropout(d2),
            Dense(1, activation='sigmoid')
        ])
        nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        nn.fit(X_train, y_train, epochs=500, batch_size=16,
               verbose=0, validation_data=(X_test, y_test))
        nn_acc = nn.evaluate(X_test, y_test, verbose=0)[1]

        nn_filename = f"{scaler_name}_nn_set{idx}.h5"
        nn.save(os.path.join(output_folder, nn_filename))

        results.append({
            'Scaler': scaler_name,
            'Base Model': 'Neural Network',
            'Set': idx,
            'Accuracy': nn_acc
        })

# Convert results to DataFrame and save CSV
df_results = pd.DataFrame(results)
csv_path = os.path.join(output_folder, "model_comparison_results.csv")
df_results.to_csv(csv_path, index=False)

# Generate detailed PDF report
pdf_path = os.path.join(output_folder, "detailed_results.pdf")
with PdfPages(pdf_path) as pdf:
    # Page 1: summary table
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    summary = df_results.groupby(['Scaler', 'Base Model'])['Accuracy'].mean().reset_index()
    summary['Accuracy (%)'] = (summary['Accuracy'] * 100).round(2)
    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    pdf.savefig(fig)
    plt.close(fig)

    # Subsequent pages: bar plots per scaler
    for sc in df_results['Scaler'].unique():
        fig, ax = plt.subplots(figsize=(8, 6))
        avg = (df_results[df_results['Scaler'] == sc]
               .groupby('Base Model')['Accuracy']
               .mean()
               .sort_values(ascending=False) * 100)
        colors = sns.color_palette("viridis", len(avg))
        ax.barh(avg.index, avg.values, color=colors, edgecolor='black')
        ax.set_xlim(0, 100)
        ax.set_title(f"Average Accuracy by Model ({sc})")
        ax.set_xlabel("Accuracy (%)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"CSV saved to: {csv_path}")
print(f"Detailed PDF saved to: {pdf_path}")
