#!/bin/bash
# Start checker_client.py in the background
python -u checker_client.py &

# Wait for it to fully start (this is a naive wait, you might want to improve it)
sleep 10

# Get its PID
pid=$(pgrep -f "python -u checker_client.py")

# Run the hypno.py script to perform injection
sudo python hypno_script.py $pid
