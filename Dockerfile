# 1. Use an official Python runtime as a parent image
FROM python:3.12-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the needed packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your code (app.py, model.py, fashion_scout_v2.pth)
COPY . .

# 6. Make port 8000 available to the world outside this container
EXPOSE 8000

# 7. Run uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]