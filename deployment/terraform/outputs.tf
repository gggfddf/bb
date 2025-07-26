# Terraform Outputs for Trading System Infrastructure

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnets
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnets
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = module.vpc.database_subnets
}

output "database_subnet_group_name" {
  description = "Name of the database subnet group"
  value       = module.vpc.database_subnet_group_name
}

# EKS Outputs
output "eks_cluster_id" {
  description = "ID of the EKS cluster"
  value       = module.eks.cluster_id
}

output "eks_cluster_arn" {
  description = "ARN of the EKS cluster"
  value       = module.eks.cluster_arn
}

output "eks_cluster_endpoint" {
  description = "Endpoint of the EKS cluster"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_version" {
  description = "Version of the EKS cluster"
  value       = module.eks.cluster_version
}

output "eks_cluster_oidc_issuer_url" {
  description = "OIDC issuer URL of the EKS cluster"
  value       = module.eks.cluster_oidc_issuer_url
}

output "eks_node_groups" {
  description = "EKS node groups"
  value       = module.eks.node_groups
}

# RDS Outputs
output "rds_cluster_id" {
  description = "ID of the RDS cluster"
  value       = module.rds.cluster_id
}

output "rds_cluster_endpoint" {
  description = "Endpoint of the RDS cluster"
  value       = module.rds.cluster_endpoint
}

output "rds_cluster_reader_endpoint" {
  description = "Reader endpoint of the RDS cluster"
  value       = module.rds.cluster_reader_endpoint
}

output "rds_cluster_port" {
  description = "Port of the RDS cluster"
  value       = module.rds.cluster_port
}

output "rds_cluster_database_name" {
  description = "Name of the RDS database"
  value       = module.rds.cluster_database_name
}

output "rds_cluster_master_username" {
  description = "Master username of the RDS cluster"
  value       = module.rds.cluster_master_username
  sensitive   = true
}

# ElastiCache Outputs
output "elasticache_cluster_id" {
  description = "ID of the ElastiCache cluster"
  value       = module.elasticache.cluster_id
}

output "elasticache_cluster_endpoint" {
  description = "Endpoint of the ElastiCache cluster"
  value       = module.elasticache.cluster_endpoint
}

output "elasticache_cluster_port" {
  description = "Port of the ElastiCache cluster"
  value       = module.elasticache.cluster_port
}

# Application Load Balancer Outputs
output "alb_id" {
  description = "ID of the Application Load Balancer"
  value       = module.alb.lb_id
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = module.alb.lb_arn
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = module.alb.lb_dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = module.alb.lb_zone_id
}

output "alb_target_group_arns" {
  description = "ARNs of the ALB target groups"
  value       = module.alb.target_group_arns
}

# Auto Scaling Group Outputs
output "asg_id" {
  description = "ID of the Auto Scaling Group"
  value       = module.asg.autoscaling_group_id
}

output "asg_arn" {
  description = "ARN of the Auto Scaling Group"
  value       = module.asg.autoscaling_group_arn
}

output "asg_name" {
  description = "Name of the Auto Scaling Group"
  value       = module.asg.autoscaling_group_name
}

output "asg_min_size" {
  description = "Minimum size of the Auto Scaling Group"
  value       = module.asg.autoscaling_group_min_size
}

output "asg_max_size" {
  description = "Maximum size of the Auto Scaling Group"
  value       = module.asg.autoscaling_group_max_size
}

output "asg_desired_capacity" {
  description = "Desired capacity of the Auto Scaling Group"
  value       = module.asg.autoscaling_group_desired_capacity
}

# ECR Outputs
output "ecr_repository_urls" {
  description = "URLs of the ECR repositories"
  value       = module.ecr.repository_urls
}

output "ecr_repository_arns" {
  description = "ARNs of the ECR repositories"
  value       = module.ecr.repository_arns
}

# Secrets Manager Outputs
output "secrets_manager_arn" {
  description = "ARN of the Secrets Manager secret"
  value       = module.secrets_manager.secret_arn
}

output "secrets_manager_name" {
  description = "Name of the Secrets Manager secret"
  value       = module.secrets_manager.secret_name
}

# CloudWatch Outputs
output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = module.cloudwatch.log_group_name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group"
  value       = module.cloudwatch.log_group_arn
}

# Route53 Outputs
output "route53_zone_id" {
  description = "ID of the Route53 hosted zone"
  value       = module.route53.zone_id
}

output "route53_zone_name" {
  description = "Name of the Route53 hosted zone"
  value       = module.route53.zone_name
}

output "route53_name_servers" {
  description = "Name servers of the Route53 hosted zone"
  value       = module.route53.name_servers
}

# WAF Outputs
output "waf_web_acl_id" {
  description = "ID of the WAF Web ACL"
  value       = module.waf.web_acl_id
}

output "waf_web_acl_arn" {
  description = "ARN of the WAF Web ACL"
  value       = module.waf.web_acl_arn
}

# Certificate Outputs
output "certificate_arn" {
  description = "ARN of the SSL certificate"
  value       = module.certificate.certificate_arn
}

output "certificate_domain_name" {
  description = "Domain name of the SSL certificate"
  value       = module.certificate.certificate_domain_name
}

# Backup Outputs
output "backup_vault_arn" {
  description = "ARN of the AWS Backup vault"
  value       = module.backup.vault_arn
}

output "backup_vault_name" {
  description = "Name of the AWS Backup vault"
  value       = module.backup.vault_name
}

# Monitoring Outputs
output "prometheus_workspace_id" {
  description = "ID of the Amazon Managed Prometheus workspace"
  value       = module.monitoring.prometheus_workspace_id
}

output "prometheus_workspace_endpoint" {
  description = "Endpoint of the Amazon Managed Prometheus workspace"
  value       = module.monitoring.prometheus_workspace_endpoint
}

output "grafana_workspace_id" {
  description = "ID of the Amazon Managed Grafana workspace"
  value       = module.monitoring.grafana_workspace_id
}

output "grafana_workspace_endpoint" {
  description = "Endpoint of the Amazon Managed Grafana workspace"
  value       = module.monitoring.grafana_workspace_endpoint
}

# Security Outputs
output "guardduty_detector_id" {
  description = "ID of the GuardDuty detector"
  value       = module.security.guardduty_detector_id
}

output "config_recorder_id" {
  description = "ID of the AWS Config recorder"
  value       = module.security.config_recorder_id
}

# Cost Optimization Outputs
output "savings_plans_arn" {
  description = "ARN of the Savings Plans"
  value       = module.cost_optimization.savings_plans_arn
}

# Disaster Recovery Outputs
output "dr_region" {
  description = "Disaster recovery region"
  value       = var.dr_region
}

output "dr_backup_vault_arn" {
  description = "ARN of the disaster recovery backup vault"
  value       = module.disaster_recovery.backup_vault_arn
}

# Application URLs
output "application_url" {
  description = "URL of the application"
  value       = "https://${var.domain_name}"
}

output "api_url" {
  description = "URL of the API"
  value       = "https://api.${var.domain_name}"
}

output "monitoring_url" {
  description = "URL of the monitoring dashboard"
  value       = "https://monitoring.${var.domain_name}"
}

# Kubernetes Configuration
output "kubeconfig" {
  description = "Kubernetes configuration for the EKS cluster"
  value       = module.eks.kubeconfig
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

# Database Connection Information
output "database_connection_string" {
  description = "Database connection string"
  value       = "postgresql://${var.database_username}:${var.database_password}@${module.rds.cluster_endpoint}:${module.rds.cluster_port}/${var.database_name}"
  sensitive   = true
}

# Redis Connection Information
output "redis_connection_string" {
  description = "Redis connection string"
  value       = "redis://${module.elasticache.cluster_endpoint}:${module.elasticache.cluster_port}"
}

# Load Balancer Health Check URL
output "health_check_url" {
  description = "URL for health checks"
  value       = "https://${var.domain_name}/health"
}

# Deployment Information
output "deployment_info" {
  description = "Deployment information"
  value = {
    environment     = var.environment
    region          = var.aws_region
    cluster_name    = var.eks_cluster_name
    domain_name     = var.domain_name
    application_url = "https://${var.domain_name}"
    api_url         = "https://api.${var.domain_name}"
    monitoring_url  = "https://monitoring.${var.domain_name}"
  }
}

# Resource Tags
output "resource_tags" {
  description = "Tags applied to all resources"
  value       = var.common_tags
}

# Cost Estimation
output "estimated_monthly_cost" {
  description = "Estimated monthly cost for the infrastructure"
  value = {
    eks_cluster     = "$500-1000"
    rds_database    = "$200-500"
    elasticache     = "$100-300"
    alb             = "$50-150"
    cloudwatch      = "$50-200"
    backup          = "$100-300"
    total           = "$1000-2450"
  }
}

# Security Information
output "security_info" {
  description = "Security information"
  value = {
    waf_enabled           = var.enable_waf
    guardduty_enabled     = var.enable_guardduty
    config_enabled        = var.enable_config
    vpc_flow_logs_enabled = var.enable_vpc_flow_logs
    backup_enabled        = var.enable_backup
    dr_enabled            = var.enable_dr
  }
}

# Compliance Information
output "compliance_info" {
  description = "Compliance information"
  value = {
    standards = var.compliance_standards
    enabled   = var.enable_compliance
  }
}