import torch
import os


# Path to the data file (from one folder back, "data/am_data.pt")
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'am_data.pt')
data_path = os.path.abspath(data_path)

# Load the data
data = torch.load(data_path)

# data = torch.load("../data/am_data.pt")

arr = data
mask = ~torch.isnan(arr).any(dim=1)

# Apply mask to filter out all rows with any NaN
arr = arr[mask]

X = arr[:, :6]   # shape: (540, 6) or (539, 6) after removing NaN sample
DIM = len(X[0])

# Targets: column 6 = Porosity, column 7 = Hardness
y_porosity = arr[:, 6]
y_hardness = arr[:, 7]
