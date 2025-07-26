# Terraform Variables for Trading System Infrastructure

# AWS Configuration
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS profile to use"
  type        = string
  default     = "default"
}

# Project Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "trading-system"
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
  default     = "production"
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "trading.example.com"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
}

variable "database_subnet_cidrs" {
  description = "CIDR blocks for database subnets"
  type        = list(string)
  default     = ["10.0.21.0/24", "10.0.22.0/24", "10.0.23.0/24"]
}

# EKS Configuration
variable "eks_cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "trading-system-cluster"
}

variable "eks_cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.27"
}

variable "eks_node_groups" {
  description = "EKS node groups configuration"
  type = map(object({
    instance_types = list(string)
    capacity_type  = string
    min_size       = number
    max_size       = number
    desired_size   = number
    disk_size      = number
  }))
  default = {
    general = {
      instance_types = ["t3.medium", "t3.large"]
      capacity_type  = "ON_DEMAND"
      min_size       = 1
      max_size       = 5
      desired_size   = 2
      disk_size      = 20
    }
    spot = {
      instance_types = ["t3.medium", "t3.large", "t3.xlarge"]
      capacity_type  = "SPOT"
      min_size       = 0
      max_size       = 10
      desired_size   = 2
      disk_size      = 20
    }
  }
}

# RDS Configuration
variable "database_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "database_allocated_storage" {
  description = "Allocated storage for RDS instance (GB)"
  type        = number
  default     = 20
}

variable "database_max_allocated_storage" {
  description = "Maximum allocated storage for RDS instance (GB)"
  type        = number
  default     = 100
}

variable "database_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "13.7"
}

variable "database_name" {
  description = "Database name"
  type        = string
  default     = "trading_system"
}

variable "database_username" {
  description = "Database master username"
  type        = string
  default     = "trading_user"
  sensitive   = true
}

variable "database_password" {
  description = "Database master password"
  type        = string
  sensitive   = true
}

variable "database_backup_retention_period" {
  description = "Database backup retention period (days)"
  type        = number
  default     = 7
}

variable "database_backup_window" {
  description = "Database backup window"
  type        = string
  default     = "03:00-04:00"
}

variable "database_maintenance_window" {
  description = "Database maintenance window"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

# ElastiCache Configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 1
}

variable "redis_parameter_group_name" {
  description = "Redis parameter group name"
  type        = string
  default     = "default.redis6.x"
}

variable "redis_port" {
  description = "Redis port"
  type        = number
  default     = 6379
}

# Application Load Balancer Configuration
variable "alb_internal" {
  description = "Whether the ALB is internal"
  type        = bool
  default     = false
}

variable "alb_idle_timeout" {
  description = "ALB idle timeout (seconds)"
  type        = number
  default     = 60
}

variable "alb_enable_deletion_protection" {
  description = "Enable ALB deletion protection"
  type        = bool
  default     = true
}

# Auto Scaling Configuration
variable "asg_min_size" {
  description = "Minimum size of Auto Scaling Group"
  type        = number
  default     = 2
}

variable "asg_max_size" {
  description = "Maximum size of Auto Scaling Group"
  type        = number
  default     = 10
}

variable "asg_desired_capacity" {
  description = "Desired capacity of Auto Scaling Group"
  type        = number
  default     = 2
}

variable "asg_health_check_grace_period" {
  description = "Health check grace period (seconds)"
  type        = number
  default     = 300
}

# Monitoring Configuration
variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch logs"
  type        = bool
  default     = true
}

variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms"
  type        = bool
  default     = true
}

variable "enable_xray" {
  description = "Enable AWS X-Ray tracing"
  type        = bool
  default     = true
}

# Security Configuration
variable "enable_vpc_flow_logs" {
  description = "Enable VPC Flow Logs"
  type        = bool
  default     = true
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty"
  type        = bool
  default     = true
}

variable "enable_config" {
  description = "Enable AWS Config"
  type        = bool
  default     = true
}

# Backup Configuration
variable "enable_backup" {
  description = "Enable AWS Backup"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention period (days)"
  type        = number
  default     = 30
}

# Tags
variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "trading-system"
    Environment = "production"
    ManagedBy   = "terraform"
    Owner       = "trading-team"
  }
}

# Container Registry Configuration
variable "ecr_repository_names" {
  description = "List of ECR repository names"
  type        = list(string)
  default = [
    "trading-system-api",
    "trading-system-worker",
    "trading-system-scheduler",
    "trading-system-monitor"
  ]
}

variable "ecr_image_tag_mutability" {
  description = "ECR image tag mutability"
  type        = string
  default     = "MUTABLE"
  validation {
    condition     = contains(["MUTABLE", "IMMUTABLE"], var.ecr_image_tag_mutability)
    error_message = "ECR image tag mutability must be either MUTABLE or IMMUTABLE."
  }
}

variable "ecr_scan_on_push" {
  description = "Enable ECR scan on push"
  type        = bool
  default     = true
}

# Secrets Manager Configuration
variable "secrets_rotation_days" {
  description = "Secrets rotation period (days)"
  type        = number
  default     = 30
}

# WAF Configuration
variable "enable_waf" {
  description = "Enable AWS WAF"
  type        = bool
  default     = true
}

variable "waf_rate_limit" {
  description = "WAF rate limit (requests per 5 minutes)"
  type        = number
  default     = 2000
}

# Route53 Configuration
variable "enable_route53" {
  description = "Enable Route53 DNS management"
  type        = bool
  default     = true
}

variable "certificate_arn" {
  description = "SSL certificate ARN"
  type        = string
  default     = ""
}

# Cost Optimization
variable "enable_cost_optimization" {
  description = "Enable cost optimization features"
  type        = bool
  default     = true
}

variable "enable_savings_plans" {
  description = "Enable AWS Savings Plans"
  type        = bool
  default     = true
}

# Disaster Recovery
variable "enable_dr" {
  description = "Enable disaster recovery"
  type        = bool
  default     = true
}

variable "dr_region" {
  description = "Disaster recovery region"
  type        = string
  default     = "us-west-2"
}

# Compliance
variable "enable_compliance" {
  description = "Enable compliance features"
  type        = bool
  default     = true
}

variable "compliance_standards" {
  description = "Compliance standards to follow"
  type        = list(string)
  default = [
    "SOC2",
    "PCI-DSS",
    "HIPAA"
  ]
}