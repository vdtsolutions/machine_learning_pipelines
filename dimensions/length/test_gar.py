import pandas as pd

REF_FILE = r"D:\Anubhav\machine_learning_pipelines\resources\Defect Sheets_12 inch.xlsx"

# Load raw Excel without header
raw = pd.read_excel(REF_FILE, header=None)

# Find header row automatically
header_row = None
for i in range(len(raw)):
    row_values = raw.iloc[i].astype(str).tolist()
    if any("Absolute Distance" in x for x in row_values):
        header_row = i
        break

print("Detected header row:", header_row)

# Load again with correct header
ref = pd.read_excel(REF_FILE, header=header_row)

# Clean column names
ref.columns = ref.columns.str.strip()

print("\nFinal detected columns:")
print(ref.columns)

# Show first rows
print("\nSample data:")
print(ref.head())

