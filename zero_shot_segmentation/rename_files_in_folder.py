import os

def rename_files_in_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.startswith("first_layer"):
            new_name = file.replace("first_layer", "firstlayer")
            old_path = os.path.join(folder_path, file)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {file} -> {new_name}")

# Example usage
folder_path = "/Users/dannybarash/Code/oct/AE_experiment/data_of_oct/"  # Update with your folder path
rename_files_in_folder(folder_path)
# folder_path = "/Users/dannybarash/Code/oct/AE_experiment/data_overfit/"  # Update with your folder path
# rename_files_in_folder(folder_path)