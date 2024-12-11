import os
import json
import pandas as pd


def test_csv_to_json_conversion():
    """
    Test the CSV to JSON conversion script.
    """

    csv_filename = "test_class_label_dictionary.csv"
    json_filename = "test_label_dictionary.json"

    # Sample data for the CSV
    data = {"label": ["Acura", "BMW", "Chevrolet"], "label_encoded": [0, 1, 2]}

    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)

    try:
        df = pd.read_csv(csv_filename)
        dictionary = df.to_dict(orient="list")

        with open(json_filename, "w") as f:
            json.dump(dictionary, f, indent=2)

        assert os.path.exists(json_filename), "JSON file was not created."

        with open(json_filename, "r") as f:
            loaded_data = json.load(f)

        assert (
            loaded_data == dictionary
        ), "The contents of the JSON file do not match the expected dictionary."

    finally:
        if os.path.exists(csv_filename):
            os.remove(csv_filename)
        if os.path.exists(json_filename):
            os.remove(json_filename)
