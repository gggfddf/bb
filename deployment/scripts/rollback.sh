#!/bin/bash

# Trading System Deployment Rollback Script
# This script provides rollback functionality for the trading system deployment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NAMESPACE="trading-system"
APP_NAME="trading-system"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

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

# Help function
show_help() {
    cat << EOF
Trading System Deployment Rollback Script

Usage: $0 [OPTIONS] [VERSION]

Options:
    -h, --help              Show this help message
    -n, --namespace NAME    Kubernetes namespace (default: trading-system)
    -a, --app-name NAME     Application name (default: trading-system)
    -k, --kubeconfig PATH   Path to kubeconfig file
    -f, --force             Force rollback without confirmation
    -d, --dry-run           Show what would be done without executing
    -v, --verbose           Enable verbose output

Examples:
    $0 v1.0.0               Rollback to version v1.0.0
    $0 -n staging v1.0.0    Rollback staging namespace to v1.0.0
    $0 -d v1.0.0            Dry run rollback to v1.0.0
    $0 -f v1.0.0            Force rollback to v1.0.0

EOF
}

# Parse command line arguments
DRY_RUN=false
FORCE=false
VERBOSE=false
TARGET_VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -a|--app-name)
            APP_NAME="$2"
            shift 2
            ;;
        -k|--kubeconfig)
            KUBECONFIG="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            TARGET_VERSION="$1"
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TARGET_VERSION" ]]; then
    log_error "Target version is required"
    show_help
    exit 1
fi

# Set kubectl context
export KUBECONFIG

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
}

# Function to check if namespace exists
check_namespace() {
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace '$NAMESPACE' does not exist"
        exit 1
    fi
}

# Function to get current deployment version
get_current_version() {
    kubectl get deployment "$APP_NAME-api" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.version}' 2>/dev/null || echo "unknown"
}

# Function to get available versions
get_available_versions() {
    kubectl get deployments -n "$NAMESPACE" -l app="$APP_NAME" -o jsonpath='{.items[*].metadata.labels.version}' 2>/dev/null | tr ' ' '\n' | sort -u
}

# Function to check if target version exists
check_target_version() {
    local version="$1"
    local available_versions
    available_versions=$(get_available_versions)
    
    if ! echo "$available_versions" | grep -q "^$version$"; then
        log_error "Target version '$version' not found"
        log_info "Available versions:"
        echo "$available_versions" | while read -r v; do
            echo "  - $v"
        done
        exit 1
    fi
}

# Function to get deployment history
get_deployment_history() {
    kubectl rollout history deployment "$APP_NAME-api" -n "$NAMESPACE" 2>/dev/null || {
        log_warning "No deployment history available"
        return 1
    }
}

# Function to rollback deployment
rollback_deployment() {
    local version="$1"
    local deployment_name="$2"
    
    log_info "Rolling back $deployment_name to version $version"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would rollback $deployment_name to version $version"
        return 0
    fi
    
    # Update deployment to target version
    kubectl patch deployment "$deployment_name" -n "$NAMESPACE" \
        -p "{\"spec\":{\"template\":{\"metadata\":{\"labels\":{\"version\":\"$version\"}}}}}" \
        --type='merge'
    
    # Wait for rollout to complete
    kubectl rollout status deployment "$deployment_name" -n "$NAMESPACE" --timeout=300s
    
    log_success "Successfully rolled back $deployment_name to version $version"
}

# Function to rollback all components
rollback_all_components() {
    local version="$1"
    
    log_info "Starting rollback of all components to version $version"
    
    # List of components to rollback
    local components=("api" "worker" "scheduler" "monitor")
    
    for component in "${components[@]}"; do
        local deployment_name="$APP_NAME-$component"
        
        # Check if deployment exists
        if kubectl get deployment "$deployment_name" -n "$NAMESPACE" &> /dev/null; then
            rollback_deployment "$version" "$deployment_name"
        else
            log_warning "Deployment $deployment_name not found, skipping"
        fi
    done
}

# Function to verify rollback
verify_rollback() {
    local version="$1"
    
    log_info "Verifying rollback to version $version"
    
    # Check deployment status
    local components=("api" "worker" "scheduler" "monitor")
    local all_healthy=true
    
    for component in "${components[@]}"; do
        local deployment_name="$APP_NAME-$component"
        
        if kubectl get deployment "$deployment_name" -n "$NAMESPACE" &> /dev/null; then
            local current_version
            current_version=$(kubectl get deployment "$deployment_name" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.version}')
            
            if [[ "$current_version" == "$version" ]]; then
                log_success "$deployment_name is running version $version"
            else
                log_error "$deployment_name is running version $current_version, expected $version"
                all_healthy=false
            fi
            
            # Check if deployment is ready
            local ready_replicas
            ready_replicas=$(kubectl get deployment "$deployment_name" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
            local desired_replicas
            desired_replicas=$(kubectl get deployment "$deployment_name" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
            
            if [[ "$ready_replicas" == "$desired_replicas" ]]; then
                log_success "$deployment_name is ready ($ready_replicas/$desired_replicas replicas)"
            else
                log_error "$deployment_name is not ready ($ready_replicas/$desired_replicas replicas)"
                all_healthy=false
            fi
        fi
    done
    
    if [[ "$all_healthy" == true ]]; then
        log_success "Rollback verification completed successfully"
    else
        log_error "Rollback verification failed"
        return 1
    fi
}

# Function to check application health
check_application_health() {
    log_info "Checking application health"
    
    # Check if API is responding
    local api_service="$APP_NAME-api"
    local api_port
    api_port=$(kubectl get service "$api_service" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "80")
    
    # Wait for service to be ready
    log_info "Waiting for API service to be ready..."
    kubectl wait --for=condition=ready pod -l app="$APP_NAME,component=api" -n "$NAMESPACE" --timeout=300s
    
    # Check health endpoint
    local health_url="http://$api_service.$NAMESPACE.svc.cluster.local:$api_port/health"
    
    if curl -f -s "$health_url" > /dev/null; then
        log_success "Application health check passed"
    else
        log_warning "Application health check failed"
        return 1
    fi
}

# Function to send notifications
send_notification() {
    local message="$1"
    local level="${2:-info}"
    
    log_info "Sending notification: $message"
    
    # Example notification (customize based on your notification system)
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"[$level] Trading System Rollback: $message\"}" \
            "$SLACK_WEBHOOK_URL" 2>/dev/null || true
    fi
    
    if [[ -n "$EMAIL_RECIPIENTS" ]]; then
        echo "$message" | mail -s "Trading System Rollback - $level" "$EMAIL_RECIPIENTS" 2>/dev/null || true
    fi
}

# Main rollback function
main_rollback() {
    local version="$1"
    
    log_info "Starting rollback process for version $version"
    log_info "Namespace: $NAMESPACE"
    log_info "Application: $APP_NAME"
    
    # Pre-rollback checks
    check_kubectl
    check_namespace
    check_target_version "$version"
    
    # Get current version
    local current_version
    current_version=$(get_current_version)
    log_info "Current version: $current_version"
    log_info "Target version: $version"
    
    # Confirmation
    if [[ "$FORCE" != true ]] && [[ "$DRY_RUN" != true ]]; then
        echo
        log_warning "This will rollback the trading system from version $current_version to version $version"
        read -p "Are you sure you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Rollback cancelled"
            exit 0
        fi
    fi
    
    # Send pre-rollback notification
    send_notification "Starting rollback from $current_version to $version" "warning"
    
    # Show deployment history
    get_deployment_history
    
    # Perform rollback
    rollback_all_components "$version"
    
    # Verify rollback
    verify_rollback "$version"
    
    # Check application health
    if check_application_health; then
        log_success "Rollback completed successfully"
        send_notification "Rollback completed successfully from $current_version to $version" "success"
    else
        log_error "Rollback completed but health check failed"
        send_notification "Rollback completed but health check failed" "error"
        exit 1
    fi
}

# Execute main function
main_rollback "$TARGET_VERSION"