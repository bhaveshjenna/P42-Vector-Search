FROM python:3.14-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
# Install exact CPU torch version first
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
# Install the rest of the requirements, ignoring torch so we don't overwrite the CPU build
RUN grep -v "torch==" requirements.txt > reqs_no_torch.txt && \
    pip install --no-cache-dir -r reqs_no_torch.txt

# Copy backend source (does not bake in the dataset since it's mounted)
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
