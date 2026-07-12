# HOWTO: Run Operations Documentation Tests

## Overview

The operations documentation includes comprehensive unit tests to validate the completeness and accuracy of the operations runbook.

## Running Tests

### Test Operations Documentation

```bash
# Run all operations documentation tests
python -m pytest tests/unit/test_operations_docs.py -v

# Run specific test categories
python -m pytest tests/unit/test_operations_docs.py::TestOperationsDocumentation -v
python -m pytest tests/unit/test_operations_docs.py::TestOperationsScriptReferences -v

# Run with coverage
python -m pytest tests/unit/test_operations_docs.py --cov=docs --cov-report=term-missing
```

### Validate Documentation Content

```bash
# Check that all required sections exist
python -c "
import re
with open('docs/operations.md', 'r') as f:
    content = f.read()

sections = [
    'Deployment Procedures',
    'Configuration Management',
    'Monitoring & Observability',
    'Troubleshooting Guide',
    'Maintenance Tasks',
    'Performance Optimization',
    'Security & Compliance',
    'Disaster Recovery'
]

for section in sections:
    if section in content:
        print(f'✅ {section}')
    else:
        print(f'❌ Missing: {section}')
"

# Validate configuration file references
python -c "
import os
files = ['config/default.yaml', 'config/development.yaml', 'config/production.yaml']
for file in files:
    if os.path.exists(file):
        print(f'✅ {file}')
    else:
        print(f'❌ Missing: {file}')
"
```

## Test Coverage

The tests validate:

- **Completeness**: All required sections are present
- **Accuracy**: Referenced files and configurations exist
- **Syntax**: Command syntax is valid
- **Structure**: Table of contents matches content
- **Metadata**: Version information and contact details
- **Procedures**: All deployment and maintenance procedures documented

## Expected Results

All tests should pass:
- 21 tests total
- TestOperationsDocumentation: 19 tests
- TestOperationsScriptReferences: 2 tests

## Troubleshooting Test Failures

### Configuration Reference Errors
```bash
# If config file references fail, ensure files exist:
ls -la config/
```

### Command Syntax Errors
```bash
# Check for unmatched quotes in bash code blocks
grep -n "python -c" docs/operations.md
```

### Missing Section Errors
```bash
# Verify section headers match table of contents
grep "^## " docs/operations.md
```

## Continuous Integration

These tests run automatically in CI/CD pipeline to ensure operations documentation remains accurate and complete.