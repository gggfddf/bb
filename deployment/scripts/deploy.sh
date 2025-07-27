#!/bin/bash

# Trading System Deployment Script
# Usage: ./deploy.sh [environment] [version]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"

# Default values
ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
NAMESPACE="trading"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        error "docker is not installed"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        warning "helm is not installed, some features may not work"
    fi
    
    success "Prerequisites check passed"
}

# Validate environment
validate_environment() {
    log "Validating environment: $ENVIRONMENT"
    
    case $ENVIRONMENT in
        development|dev)
            ENVIRONMENT="development"
            ;;
        staging|stage)
            ENVIRONMENT="staging"
            ;;
        production|prod)
            ENVIRONMENT="production"
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT"
            error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
    
    success "Environment validated: $ENVIRONMENT"
}

# Create namespace if it doesn't exist
create_namespace() {
    log "Creating namespace: $NAMESPACE"
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        kubectl create namespace "$NAMESPACE"
        success "Namespace created: $NAMESPACE"
    else
        log "Namespace already exists: $NAMESPACE"
    fi
}

# Apply Kubernetes secrets
apply_secrets() {
    log "Applying Kubernetes secrets..."
    
    # Create secrets file if it doesn't exist
    SECRETS_FILE="$DEPLOYMENT_DIR/kubernetes/secrets.yaml"
    if [ ! -f "$SECRETS_FILE" ]; then
        warning "Secrets file not found: $SECRETS_FILE"
        warning "Please create the secrets file with proper values"
        return 1
    fi
    
    kubectl apply -f "$SECRETS_FILE" -n "$NAMESPACE"
    success "Secrets applied"
}

# Apply Kubernetes configmaps
apply_configmaps() {
    log "Applying Kubernetes configmaps..."
    
    CONFIGMAP_FILE="$DEPLOYMENT_DIR/kubernetes/configmap.yaml"
    if [ -f "$CONFIGMAP_FILE" ]; then
        kubectl apply -f "$CONFIGMAP_FILE" -n "$NAMESPACE"
        success "ConfigMaps applied"
    else
        warning "ConfigMap file not found: $CONFIGMAP_FILE"
    fi
}

# Apply Kubernetes resources
apply_kubernetes_resources() {
    log "Applying Kubernetes resources..."
    
    # Apply all Kubernetes manifests
    kubectl apply -f "$DEPLOYMENT_DIR/kubernetes/" -n "$NAMESPACE"
    success "Kubernetes resources applied"
}

# Wait for deployment to be ready
wait_for_deployment() {
    log "Waiting for deployment to be ready..."
    
    DEPLOYMENT_NAME="trading-system"
    TIMEOUT=300  # 5 minutes
    
    if kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout="${TIMEOUT}s"; then
        success "Deployment is ready"
    else
        error "Deployment failed to become ready within ${TIMEOUT}s"
        return 1
    fi
}

# Run health checks
run_health_checks() {
    log "Running health checks..."
    
    # Get service port
    SERVICE_PORT=$(kubectl get service trading-system-service -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}')
    
    # Port forward to service
    kubectl port-forward service/trading-system-service 8000:"$SERVICE_PORT" -n "$NAMESPACE" &
    PF_PID=$!
    
    # Wait for port forward to be ready
    sleep 5
    
    # Run health checks
    local health_check_passed=false
    for i in {1..10}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            health_check_passed=true
            break
        fi
        log "Health check attempt $i failed, retrying..."
        sleep 5
    done
    
    # Kill port forward
    kill $PF_PID 2>/dev/null || true
    
    if [ "$health_check_passed" = true ]; then
        success "Health checks passed"
    else
        error "Health checks failed"
        return 1
    fi
}

# Rollback deployment
rollback_deployment() {
    log "Rolling back deployment..."
    
    DEPLOYMENT_NAME="trading-system"
    
    if kubectl rollout undo deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE"; then
        success "Deployment rolled back successfully"
        
        # Wait for rollback to complete
        if kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=300s; then
            success "Rollback completed"
        else
            error "Rollback failed to complete"
            return 1
        fi
    else
        error "Failed to rollback deployment"
        return 1
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill
}

# Main deployment function
deploy() {
    log "Starting deployment to $ENVIRONMENT environment..."
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Check prerequisites
    check_prerequisites
    
    # Validate environment
    validate_environment
    
    # Create namespace
    create_namespace
    
    # Apply secrets
    if ! apply_secrets; then
        error "Failed to apply secrets"
        exit 1
    fi
    
    # Apply configmaps
    apply_configmaps
    
    # Apply Kubernetes resources
    apply_kubernetes_resources
    
    # Wait for deployment
    if ! wait_for_deployment; then
        error "Deployment failed"
        rollback_deployment
        exit 1
    fi
    
    # Run health checks
    if ! run_health_checks; then
        error "Health checks failed"
        rollback_deployment
        exit 1
    fi
    
    success "Deployment completed successfully!"
}

# Show deployment status
show_status() {
    log "Showing deployment status..."
    
    echo "=== Deployment Status ==="
    kubectl get pods -n "$NAMESPACE"
    echo
    echo "=== Service Status ==="
    kubectl get services -n "$NAMESPACE"
    echo
    echo "=== Deployment Status ==="
    kubectl get deployments -n "$NAMESPACE"
    echo
    echo "=== Recent Events ==="
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

# Show logs
show_logs() {
    log "Showing application logs..."
    
    kubectl logs -f deployment/trading-system -n "$NAMESPACE"
}

# Main script logic
case "${1:-}" in
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    rollback)
        rollback_deployment
        ;;
    help|--help|-h)
        echo "Usage: $0 [environment] [version]"
        echo "       $0 status"
        echo "       $0 logs"
        echo "       $0 rollback"
        echo
        echo "Environments: development, staging, production"
        echo "Default environment: staging"
        echo "Default version: latest"
        ;;
    *)
        deploy
        ;;
esac