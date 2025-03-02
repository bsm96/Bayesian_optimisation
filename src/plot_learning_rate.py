import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the CSV file path
csv_file = "results/EI_results.csv"  # Adjusted for running from Bayesian_optimisation

# Print the current working directory for debugging
print("Current working directory:", os.getcwd())

# Check if the file exists
if not os.path.exists(csv_file):
    print(f"Error: File not found at {os.path.abspath(csv_file)}")
    exit(1)

# Load the CSV file
data = pd.read_csv(csv_file)

# Print column names for verification
print("Column names in CSV:", data.columns.tolist())

# Define the columns to plot (adjust if names differ in your CSV)
learning_rate_col = "learning_rate"
accuracy_col = "accuracy"

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data[learning_rate_col], data[accuracy_col], color='blue', label='Evaluated Points')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Accuracy (%)')
plt.title('Validation Accuracy vs. Learning Rate')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('results/learning_rate_vs_accuracy.png')  # Adjusted path
plt.show()