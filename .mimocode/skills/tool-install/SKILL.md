---
name: tool-install
description: "Automated installation of tools and plugins from GitHub repositories"
---

# Tool Installation Skill

This skill provides a standardized workflow for installing tools and plugins from GitHub repositories, with support for proxy configuration and dependency management.

## Usage

```
/tool-install <repository_url> [options]
```

### Parameters

- `repository_url`: GitHub repository URL (required)
- `--proxy <port>`: Proxy port for external access (optional)
- `--branch <branch>`: Specific branch to install (optional, default: main)
- `--force`: Force reinstall even if already installed (optional)

## Workflow

### Phase 1: Repository Analysis

1. **Validate repository URL**
   - Check URL format
   - Verify repository exists
   - Identify repository type

2. **Check existing installation**
   - Look for existing installation
   - Check version compatibility
   - Determine if update needed

### Phase 2: Environment Setup

1. **Configure network access**
   - Set up proxy if specified
   - Test connectivity
   - Handle authentication if needed

2. **Prepare installation directory**
   - Create necessary directories
   - Set appropriate permissions
   - Backup existing configurations

### Phase 3: Installation

1. **Clone repository**
   ```bash
   git clone <repository_url> <target_directory>
   ```

2. **Install dependencies**
   - Check for package.json (Node.js)
   - Check for requirements.txt (Python)
   - Check for other dependency files

3. **Configure environment**
   - Set up environment variables
   - Create configuration files
   - Initialize required services

### Phase 4: Verification

1. **Test installation**
   - Run basic commands
   - Verify functionality
   - Check for errors

2. **Integration testing**
   - Test with existing tools
   - Verify compatibility
   - Check performance

## Common Installation Patterns

### Node.js Projects
```bash
git clone <repo_url> <directory>
cd <directory>
npm install
npm run build  # if needed
```

### Python Projects
```bash
git clone <repo_url> <directory>
cd <directory>
pip install -r requirements.txt
# or
pip install -e .
```

### Plugin Systems
```bash
git clone <repo_url> <plugin_directory>
# Configure plugin in main application
# Restart application if needed
```

## Proxy Configuration

For external repository access through proxy:

```bash
# Set proxy for git
git config --global http.proxy http://localhost:<port>
git config --global https.proxy http://localhost:<port>

# Set proxy for npm
npm config set proxy http://localhost:<port>
npm config set https-proxy http://localhost:<port>

# Set proxy for pip
pip config set global.proxy http://localhost:<port>
```

## Error Handling

### Common Issues

1. **Network connectivity**
   - Check proxy configuration
   - Verify firewall settings
   - Test with curl/wget

2. **Permission errors**
   - Check directory permissions
   - Use sudo if appropriate
   - Verify user ownership

3. **Dependency conflicts**
   - Use virtual environments
   - Check version compatibility
   - Resolve dependency trees

4. **Build failures**
   - Check system requirements
   - Install build tools
   - Verify compiler versions

## Best Practices

1. **Always verify** - Test installation before considering complete
2. **Document changes** - Record what was installed and where
3. **Use isolation** - Prefer virtual environments or containers
4. **Backup first** - Save existing configurations before changes
5. **Test thoroughly** - Verify all functionality works as expected

## Output

The skill should provide:
- Installation status (success/failure)
- Installed version/location
- Any configuration changes made
- Verification results
- Next steps or recommendations