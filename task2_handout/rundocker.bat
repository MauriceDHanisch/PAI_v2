@echo off
docker build -t task2 .
docker run --rm -u 1000:1000 -v "%cd%:/results" task2
