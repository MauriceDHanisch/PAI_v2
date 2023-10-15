import sys
from hypno import inject_py

print('Hello from hypno_script.py')
if __name__ == '__main__':
    print('Running as main')

    #script_file_path = 'code.py'  # Replace with the actual path to your script file
    #with open(script_file_path, 'r') as file:
        #python_code = file.read()

    #code_to_inject = "import os; print('Hello again from', os.getpid())"  # your Python code to inject
    

    if len(sys.argv) != 3:
        print("Usage: hypno_script.py <PID> <code_to_inject>")
        sys.exit(1)

    pid = int(sys.argv[1])
    code_to_inject = sys.argv[2]  # Get the code to inject as the second command-line argument
    inject_py(pid, code_to_inject)

    
