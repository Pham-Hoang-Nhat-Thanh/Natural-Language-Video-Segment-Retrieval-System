#!/bin/bash

# Performance Testing Script for Video Retrieval System
# Tests search latency, throughput, and system performance under load

set -e

# Configuration
API_URL="${1:-http://localhost:8000}"
CONCURRENT_USERS="${2:-10}"
TEST_DURATION="${3:-60s}"
RAMP_UP_TIME="${4:-10s}"
RESULTS_DIR="./performance-results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create results directory
mkdir -p $RESULTS_DIR

# Test queries for realistic load testing
create_test_queries() {
    cat > $RESULTS_DIR/test_queries.json << 'EOF'
[
    {"query": "cats playing with toys", "top_k": 10},
    {"query": "dogs running in park", "top_k": 15},
    {"query": "children laughing and playing", "top_k": 8},
    {"query": "beautiful sunset over mountains", "top_k": 12},
    {"query": "cars driving on highway", "top_k": 20},
    {"query": "people dancing at party", "top_k": 10},
    {"query": "birds flying in sky", "top_k": 5},
    {"query": "cooking delicious food", "top_k": 15},
    {"query": "sports game highlights", "top_k": 25},
    {"query": "nature documentary scenes", "top_k": 10},
    {"query": "music concert performance", "top_k": 8},
    {"query": "family vacation memories", "top_k": 12},
    {"query": "city skyline at night", "top_k": 18},
    {"query": "ocean waves crashing", "top_k": 7},
    {"query": "winter snow activities", "top_k": 14}
]
EOF
}

# K6 load testing script
create_k6_script() {
    cat > $RESULTS_DIR/search_load_test.js << 'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// Custom metrics
const searchErrors = new Counter('search_errors');
const searchSuccessRate = new Rate('search_success_rate');
const searchLatency = new Trend('search_latency');

// Test configuration
export let options = {
    stages: [
        { duration: '${RAMP_UP_TIME}', target: ${CONCURRENT_USERS} },
        { duration: '${TEST_DURATION}', target: ${CONCURRENT_USERS} },
        { duration: '10s', target: 0 },
    ],
    thresholds: {
        'http_req_duration': ['p(95)<2000'], // 95% of requests under 2s
        'search_latency': ['p(99)<5000'],    // 99% of search requests under 5s
        'search_success_rate': ['rate>0.95'], // 95% success rate
        'search_errors': ['count<50'],        // Less than 50 errors
    },
};

// Test queries
const queries = [
    {"query": "cats playing with toys", "top_k": 10},
    {"query": "dogs running in park", "top_k": 15},
    {"query": "children laughing and playing", "top_k": 8},
    {"query": "beautiful sunset over mountains", "top_k": 12},
    {"query": "cars driving on highway", "top_k": 20},
    {"query": "people dancing at party", "top_k": 10},
    {"query": "birds flying in sky", "top_k": 5},
    {"query": "cooking delicious food", "top_k": 15},
    {"query": "sports game highlights", "top_k": 25},
    {"query": "nature documentary scenes", "top_k": 10}
];

export default function() {
    // Select random query
    const query = queries[Math.floor(Math.random() * queries.length)];
    
    // Perform search request
    const startTime = Date.now();
    const response = http.post('${API_URL}/api/v1/search', JSON.stringify(query), {
        headers: {
            'Content-Type': 'application/json',
        },
    });
    const endTime = Date.now();
    
    // Record metrics
    const latency = endTime - startTime;
    searchLatency.add(latency);
    
    // Check response
    const success = check(response, {
        'status is 200': (r) => r.status === 200,
        'response time < 5000ms': (r) => r.timings.duration < 5000,
        'has results': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.results && Array.isArray(body.results);
            } catch (e) {
                return false;
            }
        },
        'search time recorded': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.query_time_ms !== undefined;
            } catch (e) {
                return false;
            }
        }
    });
    
    searchSuccessRate.add(success);
    if (!success) {
        searchErrors.add(1);
    }
    
    // Brief pause between requests
    sleep(Math.random() * 2 + 1); // 1-3 seconds
}

export function handleSummary(data) {
    return {
        'performance_summary.json': JSON.stringify(data, null, 2),
        'performance_summary.html': htmlReport(data),
    };
}

function htmlReport(data) {
    return `
<!DOCTYPE html>
<html>
<head>
    <title>Video Search Performance Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { margin: 10px 0; padding: 10px; border: 1px solid #ddd; }
        .success { background-color: #d4edda; }
        .warning { background-color: #fff3cd; }
        .error { background-color: #f8d7da; }
    </style>
</head>
<body>
    <h1>Video Search Performance Test Results</h1>
    <h2>Test Configuration</h2>
    <p>Concurrent Users: ${CONCURRENT_USERS}</p>
    <p>Test Duration: ${TEST_DURATION}</p>
    <p>API URL: ${API_URL}</p>
    
    <h2>Key Metrics</h2>
    <div class="metric ${data.metrics.http_req_duration.values.p95 < 2000 ? 'success' : 'error'}">
        <strong>95th Percentile Response Time:</strong> ${data.metrics.http_req_duration.values.p95.toFixed(2)}ms
    </div>
    <div class="metric ${data.metrics.search_success_rate.values.rate > 0.95 ? 'success' : 'error'}">
        <strong>Success Rate:</strong> ${(data.metrics.search_success_rate.values.rate * 100).toFixed(2)}%
    </div>
    <div class="metric">
        <strong>Total Requests:</strong> ${data.metrics.http_reqs.values.count}
    </div>
    <div class="metric">
        <strong>Average Response Time:</strong> ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms
    </div>
    
    <h2>Detailed Metrics</h2>
    <pre>${JSON.stringify(data.metrics, null, 2)}</pre>
</body>
</html>
    `;
}
EOF
}

# Artillery load testing script (alternative)
create_artillery_script() {
    cat > $RESULTS_DIR/search_artillery_test.yml << 'EOF'
config:
  target: '${API_URL}'
  phases:
    - duration: ${RAMP_UP_TIME}
      arrivalRate: 1
      rampTo: ${CONCURRENT_USERS}
    - duration: ${TEST_DURATION}
      arrivalRate: ${CONCURRENT_USERS}
  payload:
    path: "./test_queries.json"
    fields:
      - "query"
      - "top_k"
  processor: "./search_processor.js"

scenarios:
  - name: "Search Videos"
    weight: 100
    flow:
      - post:
          url: "/api/v1/search"
          json:
            query: "{{ query }}"
            top_k: "{{ top_k }}"
          capture:
            - json: "$.query_time_ms"
              as: "search_time"
            - json: "$.total_results"
              as: "result_count"
          expect:
            - statusCode: 200
            - contentType: json
            - hasProperty: "results"
EOF
}

# Health check before testing
check_system_health() {
    log_info "Checking system health before testing..."
    
    # Check API Gateway health
    if curl -f "$API_URL/health" > /dev/null 2>&1; then
        log_success "API Gateway is healthy"
    else
        log_error "API Gateway health check failed"
        exit 1
    fi
    
    # Check readiness
    if curl -f "$API_URL/ready" > /dev/null 2>&1; then
        log_success "System is ready for testing"
    else
        log_warning "System readiness check failed, proceeding anyway"
    fi
}

# Run K6 performance tests
run_k6_tests() {
    log_info "Running K6 performance tests..."
    
    if ! command -v k6 &> /dev/null; then
        log_error "K6 is not installed. Please install from https://k6.io/"
        return 1
    fi
    
    cd $RESULTS_DIR
    k6 run --out json=k6_results.json search_load_test.js
    cd ..
    
    log_success "K6 tests completed"
}

# Run Artillery tests (if K6 not available)
run_artillery_tests() {
    log_info "Running Artillery performance tests..."
    
    if ! command -v artillery &> /dev/null; then
        log_error "Artillery is not installed. Please install with: npm install -g artillery"
        return 1
    fi
    
    cd $RESULTS_DIR
    artillery run search_artillery_test.yml --output artillery_results.json
    artillery report artillery_results.json --output artillery_report.html
    cd ..
    
    log_success "Artillery tests completed"
}

# Custom curl-based load test (fallback)
run_basic_load_test() {
    log_info "Running basic load test with curl..."
    
    local results_file="$RESULTS_DIR/basic_results.txt"
    local total_requests=100
    local success_count=0
    local total_time=0
    
    echo "Basic Load Test Results" > $results_file
    echo "======================" >> $results_file
    echo "Total Requests: $total_requests" >> $results_file
    echo "Concurrent Users: $CONCURRENT_USERS" >> $results_file
    echo "" >> $results_file
    
    for i in $(seq 1 $total_requests); do
        query="{\"query\": \"test query $i\", \"top_k\": 10}"
        
        start_time=$(date +%s%N)
        response=$(curl -s -w "%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -d "$query" \
            "$API_URL/api/v1/search")
        end_time=$(date +%s%N)
        
        status_code="${response: -3}"
        duration=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        
        if [ "$status_code" = "200" ]; then
            success_count=$((success_count + 1))
        fi
        
        total_time=$((total_time + duration))
        
        echo "Request $i: ${status_code} - ${duration}ms" >> $results_file
        
        # Progress indicator
        if [ $((i % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done
    
    echo ""
    
    # Calculate statistics
    avg_time=$((total_time / total_requests))
    success_rate=$((success_count * 100 / total_requests))
    
    echo "" >> $results_file
    echo "Summary:" >> $results_file
    echo "Success Rate: ${success_rate}%" >> $results_file
    echo "Average Response Time: ${avg_time}ms" >> $results_file
    echo "Total Time: ${total_time}ms" >> $results_file
    
    log_success "Basic load test completed"
    log_info "Results saved to $results_file"
}

# Analyze results
analyze_results() {
    log_info "Analyzing performance test results..."
    
    local summary_file="$RESULTS_DIR/test_summary.md"
    
    cat > $summary_file << 'EOF'
# Performance Test Summary

## Test Configuration
- **API URL**: ${API_URL}
- **Concurrent Users**: ${CONCURRENT_USERS}
- **Test Duration**: ${TEST_DURATION}
- **Date**: $(date)

## Key Findings

### Response Time Analysis
- Target: <50ms for search queries
- P95 should be under 2 seconds
- P99 should be under 5 seconds

### Throughput Analysis
- Target: Handle 1000+ concurrent users
- Success rate should be >95%

### Resource Utilization
- Monitor CPU and memory usage during tests
- Check for any bottlenecks

## Recommendations

Based on the test results:

1. **If P95 > 2s**: Consider optimizing search algorithms or scaling services
2. **If success rate < 95%**: Investigate error patterns and system stability
3. **If throughput insufficient**: Scale horizontally or optimize resource allocation

## Next Steps

1. Monitor production metrics
2. Set up automated performance regression tests
3. Implement performance budgets in CI/CD pipeline

EOF
    
    log_success "Performance analysis completed"
    log_info "Summary saved to $summary_file"
}

# Main function
main() {
    echo "Video Retrieval System - Performance Testing"
    echo "============================================"
    echo "API URL: $API_URL"
    echo "Concurrent Users: $CONCURRENT_USERS" 
    echo "Test Duration: $TEST_DURATION"
    echo ""
    
    # Create test data
    create_test_queries
    create_k6_script
    create_artillery_script
    
    # Health check
    check_system_health
    
    # Run tests (try K6 first, then Artillery, then basic)
    if command -v k6 &> /dev/null; then
        run_k6_tests
    elif command -v artillery &> /dev/null; then
        run_artillery_tests
    else
        log_warning "Neither K6 nor Artillery found, running basic load test"
        run_basic_load_test
    fi
    
    # Analyze results
    analyze_results
    
    echo ""
    log_success "Performance testing completed!"
    log_info "Results available in: $RESULTS_DIR"
    
    # Show quick summary if available
    if [ -f "$RESULTS_DIR/performance_summary.json" ]; then
        log_info "Quick Summary:"
        echo "=============="
        # Extract key metrics from JSON (requires jq)
        if command -v jq &> /dev/null; then
            echo "P95 Response Time: $(jq -r '.metrics.http_req_duration.values.p95' $RESULTS_DIR/performance_summary.json)ms"
            echo "Success Rate: $(jq -r '.metrics.search_success_rate.values.rate * 100' $RESULTS_DIR/performance_summary.json)%"
            echo "Total Requests: $(jq -r '.metrics.http_reqs.values.count' $RESULTS_DIR/performance_summary.json)"
        fi
    fi
}

# Run main function
main "$@"
