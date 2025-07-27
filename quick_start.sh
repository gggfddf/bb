#!/bin/bash

# ML Stock Predictor Platform - Quick Start Script
# This script sets up and runs the entire trading system

set -e

echo "🚀 ML Stock Predictor Platform - Quick Start"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3.9+ is required but not installed."
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    print_success "Python $python_version found"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed."
        print_warning "Please install Docker from https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is required but not installed."
        print_warning "Please install Docker Compose from https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    print_success "Docker and Docker Compose found"
}

# Setup virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Setup environment file
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Environment file created from example"
            print_warning "Please edit .env file with your configuration"
        else
            print_error ".env.example file not found"
            exit 1
        fi
    else
        print_warning "Environment file already exists"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p logs models data
    print_success "Directories created"
}

# Start Docker services
start_docker_services() {
    print_status "Starting Docker services..."
    
    # Build and start services
    docker-compose up -d --build
    
    print_success "Docker services started"
    print_status "Waiting for services to be ready..."
    
    # Wait for PostgreSQL to be ready
    print_status "Waiting for PostgreSQL..."
    while ! docker-compose exec -T postgres pg_isready -U trading_user > /dev/null 2>&1; do
        sleep 2
    done
    print_success "PostgreSQL is ready"
    
    # Wait for Redis to be ready
    print_status "Waiting for Redis..."
    while ! docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; do
        sleep 2
    done
    print_success "Redis is ready"
}

# Setup database
setup_database() {
    print_status "Setting up database..."
    
    # Run database setup script
    python scripts/setup_database.py
    
    print_success "Database setup completed"
}

# Start the application
start_application() {
    print_status "Starting the trading system application..."
    
    # Start the main application
    python scripts/run_app.py &
    APP_PID=$!
    
    print_success "Application started with PID: $APP_PID"
    print_status "Application is running in the background"
}

# Show status
show_status() {
    echo ""
    echo "🎉 ML Stock Predictor Platform is now running!"
    echo "=============================================="
    echo ""
    echo "📊 Services Status:"
    echo "  • PostgreSQL (TimescaleDB): http://localhost:5432"
    echo "  • Redis: http://localhost:6379"
    echo "  • Main Application: http://localhost:8000"
    echo "  • Prometheus (Monitoring): http://localhost:9091"
    echo "  • Grafana (Dashboard): http://localhost:3000"
    echo ""
    echo "🔑 Default Credentials:"
    echo "  • Grafana: admin / admin123"
    echo ""
    echo "📁 Important Directories:"
    echo "  • Logs: ./logs/"
    echo "  • Models: ./models/"
    echo "  • Data: ./data/"
    echo ""
    echo "🛑 To stop the application:"
    echo "  • Press Ctrl+C to stop the main application"
    echo "  • Run: docker-compose down"
    echo ""
    echo "📖 For more information, check the README.md file"
    echo ""
}

# Main execution
main() {
    echo "Starting setup process..."
    echo ""
    
    # Run all setup steps
    check_python
    check_docker
    setup_venv
    install_dependencies
    setup_env
    create_directories
    start_docker_services
    setup_database
    start_application
    show_status
    
    # Keep the script running
    wait $APP_PID
}

# Handle script interruption
cleanup() {
    print_status "Cleaning up..."
    if [ ! -z "$APP_PID" ]; then
        kill $APP_PID 2>/dev/null || true
    fi
    print_success "Cleanup completed"
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Run main function
main