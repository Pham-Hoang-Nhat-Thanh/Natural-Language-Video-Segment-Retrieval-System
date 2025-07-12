#!/bin/bash
echo "🎨 Starting Video Retrieval Frontend..."

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo "🌐 Starting Next.js development server..."
echo "🔗 Frontend will be available at: http://localhost:3000"
echo "🔗 Make sure backend services are running (run start-backend.sh or use Docker Compose)"
echo ""
echo "Press Ctrl+C to stop the server"

npm run dev
