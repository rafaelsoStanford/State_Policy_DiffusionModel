
import os
import subprocess

def visualize_folder_structure(folder_path):
    # Generate the tree output using the 'tree' command-line utility
    command = ['tree', '-L', '3', folder_path]  # Customize the depth (3 in this example)
    result = subprocess.run(command, capture_output=True, text=True)
    tree_output = result.stdout

    # Filter out specific files and folders
    filtered_output = []
    for line in tree_output.splitlines():
        if not any(pattern in line for pattern in ['.yaml', 'checkpoints', 'version', 'epoch', 'events', '__pycache__', 'pyc', 'zip', 'gif', 'jpeg', 'jpg']):
            filtered_output.append(line)

    # Print the filtered tree output
    print('\n'.join(filtered_output))

# Specify the folder path you want to visualize
folder_path = '/home/rafael/git_repos/diffusion_bare/'

# Call the function to visualize the folder structure
visualize_folder_structure(folder_path)