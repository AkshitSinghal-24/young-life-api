# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and other libraries
# Install system dependencies required for OpenCV and other libraries AND Nginx
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p temp_uploads results weights

# Configure Nginx
RUN rm /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/sites-enabled/app

# Make start script executable
RUN chmod +x start.sh

# Expose ports
EXPOSE 80
EXPOSE 8000

# Command to run the application using the start script
CMD ["./start.sh"]
