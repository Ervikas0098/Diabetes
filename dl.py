import os, urllib.request
os.makedirs('c:/Users/admin/Documents/CVPROJECT/DBTS/data', exist_ok=True)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open('c:/Users/admin/Documents/CVPROJECT/DBTS/data/diabetes_data_upload.csv', 'wb') as f:
    f.write(response.read())
print("Data downloaded successfully.")
