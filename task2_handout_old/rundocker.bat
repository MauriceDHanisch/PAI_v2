@echo off

REM Check if the Docker daemon is running
docker info > nul 2>&1
if %errorlevel% neq 0 (
    echo "Docker daemon is not running. Starting Docker..."
    
    REM Start Docker using the "dockerd" command (may vary based on your Docker installation)
    dockerd > nul 2>&1
    
    REM Wait for Docker to start (you may need to adjust the duration)
    timeout /t 10
    
    REM Verify if Docker is now running
    docker info > nul 2>&1
    if %errorlevel% neq 0 (
        echo "Failed to start Docker. Please start Docker manually and rerun the script."
        exit /b 1
    ) else (
        echo "Docker daemon has started successfully."
    )
)

REM Now that Docker is running, proceed with building and running Docker commands
docker build -t task2 .
docker run --rm -u 1000:1000 -v "%cd%:/results" task2