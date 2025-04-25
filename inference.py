import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the scaler
scaler = joblib.load('minmax_scaler.pkl')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Load the model
model = load_model('')

# New data points for testing (replace these with your actual test inputs)
new_data = np.array([
    [100, 300000, 1000],
])

# Normalize the new input data
new_data_normalized = scaler.transform(new_data)

# Make predictions
predictions = model.predict(new_data_normalized)
predicted_classes = (predictions > 0.5).astype("int32")
predicted_labels = label_encoder.inverse_transform(predicted_classes[:, 0])

# Print the results
for i in range(len(new_data)):
    input_features = ", ".join([f"P{j+1}: {new_data[i][j]}" for j in range(new_data.shape[1])])
    print(f'{input_features}, Prediction: {"Printable" if predicted_labels[i] == 1 else "Not Printable"}')
