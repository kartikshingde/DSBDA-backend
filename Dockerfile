# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY main.py .
COPY random_forest_addiction_model.pkl .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install flask scikit-learn pandas numpy flask-cors

# Expose port
EXPOSE 5000

# Command to run the app
CMD ["python", "main.py"]