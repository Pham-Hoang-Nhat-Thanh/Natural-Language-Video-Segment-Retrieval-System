#!/bin/bash

# Kubernetes Deployment Script for Video Retrieval System
# This script deploys the complete video retrieval system to a Kubernetes cluster

set -e

# Configuration
NAMESPACE="video-retrieval"
REGISTRY="ghcr.io/your-org"
VERSION="${1:-latest}"
ENVIRONMENT="${2:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check helm (optional)
    if command -v helm &> /dev/null; then
        log_info "Helm is available"
        HELM_AVAILABLE=true
    else
        log_warning "Helm is not available, using kubectl only"
        HELM_AVAILABLE=false
    fi
    
    log_success "Prerequisites check completed"
}

# Create namespace and basic resources
create_namespace() {
    log_info "Creating namespace and basic resources..."
    
    kubectl apply -f infra/k8s/00-namespace-config.yml
    
    log_success "Namespace and basic resources created"
}

# Deploy infrastructure components
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Deploy database and cache
    kubectl apply -f infra/k8s/03-infrastructure.yml
    
    # Wait for infrastructure to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgresql -n $NAMESPACE --timeout=300s
    
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s
    
    log_info "Waiting for MinIO to be ready..."
    kubectl wait --for=condition=ready pod -l app=minio -n $NAMESPACE --timeout=300s
    
    log_success "Infrastructure components deployed"
}

# Deploy application services
deploy_services() {
    log_info "Deploying application services..."
    
    # Update image tags in deployment files
    sed -i "s|:latest|:$VERSION|g" infra/k8s/01-frontend-api.yml
    sed -i "s|:latest|:$VERSION|g" infra/k8s/02-ml-services.yml
    
    # Deploy ML services (backend)
    kubectl apply -f infra/k8s/02-ml-services.yml
    
    # Wait for ML services to be ready
    log_info "Waiting for ML services to be ready..."
    kubectl wait --for=condition=ready pod -l app=ingest-service -n $NAMESPACE --timeout=600s
    kubectl wait --for=condition=ready pod -l app=search-service -n $NAMESPACE --timeout=600s
    
    # Deploy frontend and API gateway
    kubectl apply -f infra/k8s/01-frontend-api.yml
    
    # Wait for frontend services to be ready
    log_info "Waiting for frontend services to be ready..."
    kubectl wait --for=condition=ready pod -l app=api-gateway -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=frontend -n $NAMESPACE --timeout=300s
    
    log_success "Application services deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    kubectl apply -f infra/k8s/04-monitoring.yml
    
    # Wait for monitoring services
    log_info "Waiting for monitoring services to be ready..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=grafana -n $NAMESPACE --timeout=300s
    
    log_success "Monitoring stack deployed"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Create a temporary job for migrations
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration-$(date +%s)
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: migration
        image: $REGISTRY/video-retrieval-ingest:$VERSION
        command: ["python", "-c", "from database import IngestionDatabase; import asyncio; asyncio.run(IngestionDatabase().migrate())"]
        env:
        - name: DB_HOST
          value: "postgresql-service"
        - name: DB_PORT
          value: "5432"
        - name: DB_NAME
          value: "video_retrieval"
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: DB_USER
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: DB_PASSWORD
      restartPolicy: Never
EOF
    
    log_success "Database migrations completed"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check all pods are running
    log_info "Checking pod status..."
    kubectl get pods -n $NAMESPACE
    
    # Check services
    log_info "Checking service status..."
    kubectl get services -n $NAMESPACE
    
    # Check ingress
    log_info "Checking ingress status..."
    kubectl get ingress -n $NAMESPACE
    
    # Test health endpoints
    log_info "Testing health endpoints..."
    
    # Port forward for testing (in background)
    kubectl port-forward -n $NAMESPACE svc/api-gateway-service 8000:8000 &
    PF_PID=$!
    
    sleep 5
    
    # Test API gateway health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API Gateway health check passed"
    else
        log_warning "API Gateway health check failed"
    fi
    
    # Clean up port forward
    kill $PF_PID 2>/dev/null || true
    
    log_success "Deployment verification completed"
}

# Display access information
show_access_info() {
    log_info "Deployment completed successfully!"
    echo ""
    echo "Access Information:"
    echo "==================="
    
    # Get ingress IP/hostname
    INGRESS_IP=$(kubectl get ingress video-retrieval-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    INGRESS_HOST=$(kubectl get ingress video-retrieval-ingress -n $NAMESPACE -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "not-configured")
    
    echo "Frontend URL: https://$INGRESS_HOST"
    echo "API Gateway: https://$INGRESS_HOST/api"
    
    if [ "$INGRESS_IP" != "pending" ] && [ "$INGRESS_IP" != "" ]; then
        echo "Ingress IP: $INGRESS_IP"
    fi
    
    echo ""
    echo "Port Forwarding Commands:"
    echo "========================="
    echo "Frontend: kubectl port-forward -n $NAMESPACE svc/frontend-service 3000:3000"
    echo "API Gateway: kubectl port-forward -n $NAMESPACE svc/api-gateway-service 8000:8000"
    echo "Grafana: kubectl port-forward -n $NAMESPACE svc/grafana-service 3001:3000"
    echo "Prometheus: kubectl port-forward -n $NAMESPACE svc/prometheus-service 9090:9090"
    
    echo ""
    echo "Monitoring Credentials:"
    echo "======================"
    echo "Grafana - admin:admin123"
    
    echo ""
    echo "Useful Commands:"
    echo "================"
    echo "Watch pods: kubectl get pods -n $NAMESPACE -w"
    echo "View logs: kubectl logs -f deployment/<service-name> -n $NAMESPACE"
    echo "Scale service: kubectl scale deployment <service-name> --replicas=<count> -n $NAMESPACE"
}

# Cleanup function
cleanup() {
    log_warning "Cleaning up deployment..."
    
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    
    log_success "Cleanup completed"
}

# Main deployment flow
main() {
    echo "Video Retrieval System - Kubernetes Deployment"
    echo "=============================================="
    echo "Version: $VERSION"
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo ""
    
    case "${3:-deploy}" in
        "deploy")
            check_prerequisites
            create_namespace
            deploy_infrastructure
            deploy_services
            deploy_monitoring
            run_migrations
            verify_deployment
            show_access_info
            ;;
        "cleanup")
            cleanup
            ;;
        "verify")
            verify_deployment
            ;;
        *)
            echo "Usage: $0 [version] [environment] [action]"
            echo "Actions: deploy (default), cleanup, verify"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
