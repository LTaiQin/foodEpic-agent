---
description: "Install tools and plugins from GitHub repositories with proxy support"
agent: "tool-install"
---

# Tool Installation Command

Please install the tool/plugin from the specified GitHub repository: $ARGUMENTS

## Installation Instructions

1. **Parse the repository URL** - Extract repository information
2. **Check existing installation** - Verify if already installed
3. **Configure network access** - Set up proxy if needed
4. **Install dependencies** - Handle all required packages
5. **Verify installation** - Test functionality

## Common Parameters

The arguments may include:
- Repository URL (required)
- Proxy port (e.g., "--proxy 7890")
- Branch name (e.g., "--branch develop")
- Force reinstall (e.g., "--force")

## Installation Steps

### 1. Repository Analysis
```bash
# Validate repository URL
# Check repository type and requirements
# Identify installation method
```

### 2. Environment Setup
```bash
# Configure proxy if specified
# Set up installation directory
# Prepare dependencies
```

### 3. Installation Process
```bash
# Clone repository
# Install dependencies
# Configure environment
# Build if necessary
```

### 4. Verification
```bash
# Test basic functionality
# Verify integration
# Check for errors
```

## Error Handling

Handle common issues:
- Network connectivity problems
- Permission errors
- Dependency conflicts
- Build failures

## Output

Provide:
- Installation status
- Installed location
- Configuration changes
- Verification results
- Next steps