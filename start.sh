#!/bin/bash

# Start Nginx
echo "Starting Nginx..."
nginx

# Start Uvicorn
echo "Starting Uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
