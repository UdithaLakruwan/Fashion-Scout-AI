# Fashion Scout AI 
An End-to-End Computer Vision API built using the Data Science Hierarchy of Needs.

## Tech Stack
* **Level 1 (Data):** FashionMNIST Dataset
* **Level 2 (Clean):** PIL & NumPy for image normalization/inversion
* **Level 3 (Logic):** PyTorch CNN (Convolutional Neural Network)
* **Level 4 (DevOps):** FastAPI, Docker, and Ngrok for tunneling
* **Level 5 (Value):** Publicly accessible API for clothing classification

## Run with Docker
To run this project locally without installing Python dependencies:
```bash
docker build -t fashion-scout-ai .
docker run -p 8000:8000 fashion-scout-ai
