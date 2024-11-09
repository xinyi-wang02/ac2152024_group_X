import pandas as pd
import json
df = pd.read_csv("car_preprocessed_folder_class_label_dictionary.csv")
dictionary = df.to_dict()
with open("label_dictionary.json", "w") as f:
    json.dump(dictionary, f, indent=2)