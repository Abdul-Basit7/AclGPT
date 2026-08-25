.PHONY: help install install-backend install-frontend dev backend frontend build test lint clean

BACKEND_PY := backend/.venv/bin/python

help:
	@echo "make install    Install backend and frontend dependencies"
	@echo "make dev        Run the API and the Vite dev server together"
	@echo "make backend    Run only the API on :8000"
	@echo "make frontend   Run only the Vite dev server on :5173"
	@echo "make build      Build the frontend into frontend/dist"
	@echo "make test       Run the backend test suite"
	@echo "make clean      Remove build output, caches and local runtime data"

install: install-backend install-frontend

install-backend:
	python3 -m venv backend/.venv
	$(BACKEND_PY) -m pip install --upgrade pip
	$(BACKEND_PY) -m pip install -r backend/requirements-dev.txt

install-frontend:
	cd frontend && npm install

dev:
	@echo "Sourcery: API on http://127.0.0.1:8000  ·  UI on http://localhost:5173"
	@trap 'kill 0' EXIT INT TERM; \
	( cd backend && .venv/bin/python -m uvicorn app.main:app --reload --port 8000 ) & \
	( cd frontend && npm run dev ) & \
	wait

backend:
	cd backend && .venv/bin/python -m uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

build:
	cd frontend && npm run build

test:
	cd backend && .venv/bin/python -m pytest -q

lint:
	cd frontend && npm run typecheck

clean:
	rm -rf frontend/dist frontend/node_modules/.vite backend/.pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
