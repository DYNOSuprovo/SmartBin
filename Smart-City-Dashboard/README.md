# Smart City Waste Management Dashboard

This directory contains the web-based monitoring system for city-wide smart bins.

## Project Structure
- **backend/**: FastAPI server that manages bin state and provides REST APIs.
- **frontend/**: Vite/React application featuring a live map and IoT bin control panel.

## How to Run

### 1. Prerequisites
Ensure you have the Python virtual environment and Node.js installed.

### 2. Start the Backend
From the root of the `SmartBin` repository:
```bash
cd Smart-City-Dashboard/backend
..\\..\\.venv\\Scripts\\python -m uvicorn main:app --reload --port 8000
```
The API will be available at [http://localhost:8000](http://localhost:8000).

### 3. Start the Frontend
From the root of the `SmartBin` repository:
```bash
cd Smart-City-Dashboard/frontend
npm install   # If running for the first time
npm run dev
```
The dashboard will be available at [http://localhost:5173](http://localhost:5173).

## Key Features
- **Live Map**: REAL-TIME monitoring of bin levels across the city.
- **IoT Control**: Simulate bin fill levels for testing and demonstration.
- **Collection Routing**: Automated truck routing to collect waste from full bins (threshold improved to 500m for reliability).
