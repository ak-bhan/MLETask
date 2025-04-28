# MLE Take-Home Assignment Starter

## Setup

```bash
docker-compose up --build
```

Service runs on http://localhost:8000


# API Endpoints

## Iris Model (Flower Classification)

- **POST /data** – Add a data sample
- **GET /data** – List all samples (from mongodb and from newly added sample)
- **POST /train** – Train the Iris model
- **POST /predict** – Predict species from input features

---

## CPS Model (News Headline Classification)

- **POST /predict-cps** – Predict category from news headline

