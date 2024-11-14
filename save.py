import os
import sys

def save_code_to_txt(root_dir, output_file):
    try:
        with open(output_file, 'w') as f:
            for dirpath, dirnames, filenames in os.walk(root_dir):
                # Skip 'venv' directories and hidden directories
                dirnames[:] = [d for d in dirnames if d != 'venv' and not d.startswith('.')]
                for filename in filenames:
                    if filename.endswith('.py'):
                        file_path = os.path.join(dirpath, filename)
                        relative_path = os.path.relpath(file_path, root_dir)

                        # Write the relative path of the file
                        f.write(f"{relative_path}\n")
                        f.write("code\n")

                        try:
                            # Write the content of the file
                            with open(file_path, 'r', encoding='utf-8') as code_file:
                                f.write(code_file.read())
                        except Exception as e:
                            print(f"Error reading file {file_path}: {str(e)}")
                            continue

                        f.write("\n" + "-" * 80 + "\n")
    except Exception as e:
        print(f"Error saving code to file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Get the absolute path of the project root
    project_root = '/Users/dmpantiu/PycharmProjects/hackathon_hamburg/climsight'
    
    # Verify the directory exists
    if not os.path.exists(project_root):
        print(f"Error: Directory {project_root} does not exist")
        sys.exit(1)

    # Create output file path
    output_file = os.path.join(project_root, 'project_structure.txt')

    print(f"Scanning directory: {project_root}")
    save_code_to_txt(project_root, output_file)
    print(f"Project structure and code saved to {output_file}")
