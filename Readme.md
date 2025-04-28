# MLE Take-Home Assignment Starter

## Setup

```bash
docker-compose up --build
```

Service runs on http://localhost:8000


## Endpoints
POST /data – Add a data sample { "text": "...", "label": "..." }

GET /data?label=... – List samples

POST /train – Train the model

POST /predict – Predict label from input { "text": "..." }