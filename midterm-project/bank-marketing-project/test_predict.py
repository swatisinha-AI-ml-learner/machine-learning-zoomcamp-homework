"""Send a sample customer to the running service.

Usage:
    python test_predict.py                      # local service
    python test_predict.py https://your-url     # deployed service
"""
import sys

import requests

base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696"

customer = {
    "age": 42,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "success",
}

response = requests.post(f"{base_url}/predict", json=customer, timeout=30)
print(response.status_code, response.json())
