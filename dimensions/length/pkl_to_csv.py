import pandas as pd

# Load pickle file
df = pd.read_pickle(r"D:\Anubhav\machine_learning_pipelines\resources\12\data\old_data_2025\t\GMFL_12inch_24-01-2025_PTT(1)\3.pkl")

# Save to CSV
df.to_csv("data.csv", index=False)
