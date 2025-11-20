import os

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    if not lines:
        print(f"Warning: {file_path} is empty.")
        return

    first_line = lines[0].strip()

    # Determine the type based on the first line's content
    if ',' in first_line:
        new_first_line = "Final split ratio: 0.998,0.002\n"
    elif ';' in first_line:
        new_first_line = "Final split ratio: 0.998;0.002\n"
    else:
        print(f"Warning: {file_path} does not contain a comma or semicolon in the first line.")
        return

    # Update the lines with the new first line
    updated_lines = [new_first_line] + lines

    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(updated_lines)

def main():
    directory = '.'  # Assume files are in the current directory
    for i in range(1, 301):
        file_name = f"spatial{i:03}.txt"
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            process_file(file_path)
        else:
            print(f"Warning: {file_path} does not exist.")

if __name__ == "__main__":
    main()
