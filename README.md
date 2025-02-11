# 🍷 Wine Origin Prediction Service

A machine learning service that predicts the **country of origin** of wines based on their **description, price, points, and variety**.

---

## 📌 Project Overview

This project consists of:
1. **A machine learning model** trained on a wine review dataset.
2. **A FastAPI backend service** for inference and API handling.
3. **A Next.js frontend application** to interact with the API.

---

## 🏗️ Architecture

```
AWS Cloud
┌──────────────┐      ┌───────────────┐      ┌──────────────┐      ┌──────────────┐
│              │      │  API Gateway  │      │    Lambda    │      │ ECS/Fargate  │
│   Client     ├─────►│ + Custom      ├─────►│ (Request     ├─────►│ (ML Model    │
│              │      │  Domain       │      │  Handler)    │      │  Service)    │
└──────────────┘      └───────────────┘      └──────────────┘      └──────────┬───┘
                                                                          │    
┌──────────▼───┐
│  ECR Image   │
│ Repository   │
└──────────────┘

┌───────────────┐      ┌──────────────┐      ┌──────────────┐
│  CloudWatch   │      │  S3 Bucket   │      │  CloudFront  │
│  (Logging)    │      │ (ML Models)  │      │ (Cache)      │
└───────────────┘      └──────────────┘      └──────────────┘
```

---

## 🔹 **Architecture Components**

### **1️⃣ Frontend (CloudFront + S3)**
- **Static assets** served through **CloudFront CDN**.
- **React/Next.js application** hosted in **S3**.
- **Global distribution** with low latency.

### **2️⃣ API Layer (API Gateway + Lambda)**
- RESTful API endpoints.
- Request validation and authentication.
- Rate limiting and throttling.
- Custom domain and SSL/TLS.

### **3️⃣ ML Service (ECS/Fargate)**
- **Containerized ML model service**.
- **Auto-scaling** based on demand.
- **GPU support** if needed.
- **Cost-effective** for consistent workloads.

### **4️⃣ Storage & Caching**
- **S3** for **model artifacts**.
- **CloudFront** for **response caching**.
- **ECR** for **Docker images**.

### **5️⃣ Monitoring & Logging**
- **CloudWatch** for **metrics and logs**.
- **Alarms** for **error rates**.
- **Performance monitoring**.

---

## 🚀 **Setup & Local Development**

### **Prerequisites**
- Python **3.8+**

### **Backend Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scriptsctivate on Windows

# Install dependencies
cd api
pip install -r requirements.txt

# Start the API
uvicorn main:app --reload
```

---
## 🔄 **API Usage**

### **Example API Request (POST)**
```json
{
  "description": "A rich and full-bodied wine with notes of black cherry and vanilla",
  "points": 92,
  "price": 45.0,
  "variety": "Cabernet Sauvignon"
}
```

### **Example API Response**
```json
{
  "predicted_country": "France",
  "confidence_scores": {
    "France": 0.85,
    "Italy": 0.10,
    "Spain": 0.03,
    "US": 0.02
  }
}
```
---
## 🏗️ **Deployment Strategy**

### **Why AWS Lambda / Fargate?**
- **AWS Lambda** (Serverless) handles low-latency requests cost-effectively.
- **AWS Fargate** (Managed Containers) runs a persistent ML model API with auto-scaling.

### **Deployment Process**
1. **Dockerization**: The FastAPI application is **containerized using Docker**.
2. **Push to AWS ECR**: The container is uploaded to **Amazon Elastic Container Registry (ECR)**.
3. **CI/CD Pipeline**: **GitHub Actions** or **AWS CodePipeline** handles automated deployments.
4. **Infrastructure as Code (IaC)**: AWS **CloudFormation/Terraform** for managing deployment.

### **Production Considerations**
- **Load balancing** via **AWS Application Load Balancer (ALB)**.
- **Auto-scaling** based on API traffic.
- **Authentication** via **Cognito/Auth0**.

