docker build --tag task1 .
docker run --rm -v "%cd%:/results" task1
