#!/usr/bin/env python3
"""
Test runner for the video retrieval system.
Runs all tests with proper setup and teardown.
"""

import subprocess
import sys
import time
import docker
import requests
from pathlib import Path

def wait_for_service(url: str, timeout: int = 60) -> bool:
    """Wait for a service to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False

def setup_test_environment():
    """Setup the test environment."""
    print("🔧 Setting up test environment...")
    
    # Start test services with Docker Compose
    subprocess.run([
        "docker", "compose", "-f", "docker-compose.test.yml", "up", "-d"
    ], check=True)
    
    # Wait for services to be ready
    services = [
        ("http://localhost:8000", "API Gateway"),
        ("http://localhost:8001", "Ingestion Service"),
        ("http://localhost:8002", "Search Service")
    ]
    
    for url, name in services:
        print(f"⏳ Waiting for {name}...")
        if not wait_for_service(url):
            print(f"❌ {name} failed to start")
            return False
        print(f"✅ {name} is ready")
    
    return True

def run_tests():
    """Run all tests."""
    print("🧪 Running tests...")
    
    # Run pytest with coverage
    result = subprocess.run([
        "python", "-m", "pytest",
        "tests/",
        "-v",
        "--cov=backend",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--junitxml=test-results.xml"
    ])
    
    return result.returncode == 0

def cleanup_test_environment():
    """Cleanup the test environment."""
    print("🧹 Cleaning up test environment...")
    subprocess.run([
        "docker", "compose", "-f", "docker-compose.test.yml", "down", "-v"
    ])

def main():
    """Main test runner."""
    project_root = Path(__file__).parent.parent
    
    try:
        # Change to project root
        subprocess.run(["cd", str(project_root)], shell=True)
        
        # Setup environment
        if not setup_test_environment():
            print("❌ Failed to setup test environment")
            sys.exit(1)
        
        # Run tests
        success = run_tests()
        
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test run interrupted")
        success = False
    except Exception as e:
        print(f"❌ Test run failed: {e}")
        success = False
    finally:
        # Always cleanup
        cleanup_test_environment()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
