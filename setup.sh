#!/bin/bash

echo "======================================================"
echo " SignLearn: NSL Project (Linux/Mac)"
echo "======================================================"

echo "[1/4] Building and starting Docker containers..."
docker-compose up --build -d

echo ""
echo "[2/4] Waiting for PostgreSQL to initialize (15 seconds)..."
sleep 15

echo ""
echo "[3/4] Running Alembic Migrations..."
docker exec -it -w /app/backend nsl_backend_container alembic upgrade head

echo ""
echo "[4/4] Seeding Database (Avatars, Signs, Badges)..."
docker exec -it -w /app/backend nsl_backend_container python seed_data.py

echo ""
echo "======================================================"
echo "✅ Setup Complete!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔌 Backend API: http://localhost:8000/docs"
echo "======================================================"
