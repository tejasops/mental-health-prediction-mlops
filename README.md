# Mental Health Treatment Prediction — MLOps Pipeline

A production-grade MLOps project that predicts mental health treatment recommendations using machine learning, deployed with a full DevOps/MLOps stack.

---

## Architecture

```
              Developer
                  │
                  │  git push
                  ▼
        GitHub Actions CI
        ├── Run pytest (6 tests)
        └── Build & push Docker image
                  │
                  ▼
             Docker Hub
      (tejasops/mental-health-api)
                  │
                  │  auto-sync
                  ▼
           ArgoCD GitOps
                  │
                  │  applies Helm chart
                  ▼
         Kubernetes Cluster
                  │
          ┌───────┴───────────────┐
          │      Helm Chart       │
          │  Deployment (2 pods)  │
          │  Service (ClusterIP)  │
          │  Ingress (nginx)      │
          └───────┬───────────────┘
                  │
                  ▼
           Flask REST API
           ├── POST /predict   → ML prediction
           ├── GET  /health    → K8s probe
           ├── GET  /api       → Service info
           ├── GET  /          → Web UI
           └── GET  /metrics   → Prometheus
                  │
                  │  scrape every 15s
                  ▼
             Prometheus
      (request rate, latency, memory, CPU)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | scikit-learn (Random Forest) |
| API | Flask + Gunicorn |
| Containerization | Docker (multi-stage build) |
| Orchestration | Kubernetes |
| Package Manager | Helm |
| CI/CD | GitHub Actions |
| GitOps | ArgoCD |
| Monitoring | Prometheus + prometheus-flask-exporter |
| Infrastructure | Minikube |

---

## Project Structure

```
mental-health-prediction-mlops/
├── app/
│   ├── app.py                       # Flask REST API
│   └── templates/                   # Web UI
├── src/                             # ML pipeline modules
├── models/                          # Trained model artifacts
├── data/                            # Raw and processed data
├── notebooks/                       # EDA and modeling notebooks
├── tests/
│   └── test_app.py                  # pytest suite (6 tests)
├── mental-health-api/               # Helm chart
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
├── k8s/                             # Raw Kubernetes manifests
├── .github/workflows/ci.yml         # GitHub Actions CI pipeline
├── argocd-app.yaml                  # ArgoCD Application manifest
├── prometheus-scrape-config.yaml
├── Dockerfile
├── train.py
└── requirements.txt
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api` | GET | Service info (JSON) |
| `/health` | GET | Liveness/readiness probe |
| `/predict` | POST | ML prediction |
| `/metrics` | GET | Prometheus metrics |

### Prediction Request

```bash
curl -X POST http://<host>/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 28, "Gender": "male", "family_history": "Yes", "work_interfere": "Often"}'
```

### Prediction Response

```json
{
  "prediction": "Yes",
  "recommends_treatment": true
}
```

---

## CI/CD Pipeline

### GitHub Actions
1. **Test job** — runs pytest on every push/PR to main
2. **Build & Push job** — builds Docker image and pushes to Hub on main push if tests pass

### ArgoCD GitOps
- Auto-syncs Helm chart changes to cluster within 3 minutes
- `selfHeal: true` — reverts manual cluster changes
- `prune: true` — removes resources deleted from Git

---

## Monitoring

Prometheus scrapes `/metrics` every 15 seconds.

```promql
# Request rate
rate(flask_http_request_total[5m])

# Average response time
rate(flask_http_request_duration_seconds_sum[5m]) / rate(flask_http_request_duration_seconds_count[5m])

# Error rate
rate(flask_http_request_total{status=~"5.."}[5m])
```

---

## Local Development

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -v
python app/app.py
```

---

## Docker

```bash
docker pull tejasops/mental-health-api:latest
docker run -p 5000:5000 tejasops/mental-health-api:latest
```

---

## Author

**Tejas Allada**
B.Tech CSE (AIML) — Theem College of Engineering, Mumbai
[LinkedIn](https://linkedin.com/in/tejasallada) | [GitHub](https://github.com/tejasops)
