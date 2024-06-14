FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory
WORKDIR /app

# Copy the script into the container
COPY scoring_script.py .

# Install the required packages
RUN pip install pandas pyarrow scikit-learn

# Set the entrypoint to the script
ENTRYPOINT ["python", "scoring_script.py"]
