import os
import requests

DIAL_API_KEY = os.getenv("DIAL_API_KEY")
BASE = "https://ai-proxy.lab.epam.com"

r = requests.get(
    #f"{BASE}/openai/models",
    f"{BASE}/openai/deployments",
    headers={"api-key": DIAL_API_KEY},
    timeout=30,
)

print("status:", r.status_code)
print(r.text)