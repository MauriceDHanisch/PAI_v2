#!/bin/bash
# Start checker_client.py in the background
python -u checker_client.py &

# Wait for it to fully start (this is a naive wait, you might want to improve it)
echo
echo
echo "Waiting for checker_client.py to start..."
sleep 120
echo "Done waiting."

# Get its PID
pid=$(pgrep -f "python -u checker_client.py")
echo
echo "checker_client.py PID is $pid"

# Read the contents of code.py into a variable
echo
echo "Reading code.py..."
code_contents=$(cat code.py)

# Run the hypno_script.py script with code_contents as an argument
echo
echo "Running hypno_script.py..."
python hypno_script.py $pid "$code_contents"

# Run the hypno.py script to perform injection
#python hypno_script.py $pid
