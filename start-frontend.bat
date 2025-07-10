@echo off
echo 🎨 Starting Video Retrieval Frontend...

cd frontend

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
)

echo 🌐 Starting Next.js development server...
echo 🔗 Frontend will be available at: http://localhost:3000
echo 🔗 Make sure backend services are running (run start-backend.bat or start-system.bat)
echo.
echo Press Ctrl+C to stop the server

npm run dev
