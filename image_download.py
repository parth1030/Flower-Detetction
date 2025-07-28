'''
ONLY RUN ONCE!!!!!!
'''

import kagglehub

# Download latest version
path = kagglehub.dataset_download("kacpergregorowicz/house-plant-species")

print("Path to dataset files:", path)

