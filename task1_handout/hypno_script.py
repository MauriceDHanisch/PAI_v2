import sys
from hypno import inject_py

if __name__ == '__main__':
    pid = int(sys.argv[1])
    code_to_inject = '...'  # your Python code to inject
    inject_py(pid, code_to_inject)
