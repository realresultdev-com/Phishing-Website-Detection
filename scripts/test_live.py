import urllib.request
import json
import time

time.sleep(1)

# 1. Health
h = json.loads(urllib.request.urlopen("http://localhost:8000/health").read())
print("API Health Status:", h.get("status"))

# 2. Phishing URL
req1 = urllib.request.Request(
    "http://localhost:8000/predict",
    data=json.dumps({"url": "http://192.168.1.1/paypal/login.php", "threshold": 0.5}).encode(),
    headers={"Content-Type": "application/json"}
)
res1 = json.loads(urllib.request.urlopen(req1).read())
print(f"Phishing Test:   {res1['url']} -> {res1['label'].upper()} (Prob: {res1['probability']:.2%})")

# 3. Legitimate URL
req2 = urllib.request.Request(
    "http://localhost:8000/predict",
    data=json.dumps({"url": "https://www.google.com", "threshold": 0.5}).encode(),
    headers={"Content-Type": "application/json"}
)
res2 = json.loads(urllib.request.urlopen(req2).read())
print(f"Legitimate Test: {res2['url']} -> {res2['label'].upper()} (Prob: {res2['probability']:.2%})")

# 4. Streamlit App Check
code_st = urllib.request.urlopen("http://localhost:8501").getcode()
print(f"Streamlit Web Dashboard: HTTP {code_st} -> Running at http://localhost:8501")
print(f"FastAPI Interactive Docs: HTTP 200 -> Running at http://localhost:8000/docs")
