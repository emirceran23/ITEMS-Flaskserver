import pickle

def pickle_to_string(file_path):
    # Open the pickle file in binary read mode
    with open(file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)
    # Return the string representation of the data
    return str(data)

if __name__ == "__main__":
    file_path = "calibration.pickle"  # Replace with your pickle file path
    try:
        result = pickle_to_string(file_path)
        print("Decompiled content:")
        print(result)
    except Exception as error:
        print("Error during decompiling:", error)
