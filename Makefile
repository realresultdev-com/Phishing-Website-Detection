.PHONY: setup data train evaluate explain app api test clean

PYTHON=python

setup:
	pip install -r requirements.txt
	$(PYTHON) -c "import tldextract; tldextract.TLDExtract(cache_dir='.tld_cache')(url='example.com')" || true
	@echo "Setup complete."

data:
	$(PYTHON) scripts/download_data.py || $(PYTHON) scripts/make_sample_data.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

explain:
	$(PYTHON) scripts/explain_report.py

app:
	streamlit run src/app/streamlit_app.py --server.port 8501

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache htmlcov
