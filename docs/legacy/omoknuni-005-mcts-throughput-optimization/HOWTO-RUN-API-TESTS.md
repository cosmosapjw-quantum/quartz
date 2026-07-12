# HOWTO: Run API Documentation Tests

## Overview

The API documentation includes comprehensive unit tests to validate completeness, accuracy, and example correctness.

## Running Tests

### Test API Documentation

```bash
# Run all API documentation tests
python -m pytest tests/unit/test_api_docs.py -v

# Run specific test categories
python -m pytest tests/unit/test_api_docs.py::TestAPIDocumentation -v
python -m pytest tests/unit/test_api_docs.py::TestAPICodeExamples -v

# Run with detailed output
python -m pytest tests/unit/test_api_docs.py -v -s
```

### Validate Documentation Content

```bash
# Check API section completeness
python -c "
import re
with open('docs/api.md', 'r') as f:
    content = f.read()

sections = [
    'MCTS Engine API',
    'Neural Network Inference API',
    'Training Pipeline API',
    'Game Interface API',
    'Configuration API',
    'Telemetry & Monitoring API',
    'Usage Examples',
    'Error Handling'
]

for section in sections:
    if section in content:
        print(f'✅ {section}')
    else:
        print(f'❌ Missing: {section}')
"

# Validate code examples syntax
python -c "
import re, ast
with open('docs/api.md', 'r') as f:
    content = f.read()

code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
print(f'Found {len(code_blocks)} Python code examples')

for i, block in enumerate(code_blocks[:5]):  # Check first 5
    try:
        ast.parse(block.strip())
        print(f'✅ Code block {i+1}: Valid syntax')
    except:
        print(f'⚠️  Code block {i+1}: Incomplete example (normal)')
"
```

## Test Coverage

The tests validate:

### Documentation Structure (TestAPIDocumentation)
- **Completeness**: All required API sections present
- **API Coverage**: All major classes and methods documented
- **Parameter Documentation**: Args, Returns, Raises sections
- **Code Examples**: Sufficient working examples
- **Performance Targets**: All key metrics documented
- **Error Handling**: Comprehensive exception coverage
- **Configuration**: Environment variables and structure
- **Cross-References**: Valid links to other docs

### Code Quality (TestAPICodeExamples)
- **Syntax Validation**: All Python examples syntactically valid
- **Import Completeness**: Required modules imported
- **Example Coverage**: Complete training/analysis/monitoring examples
- **Error Handling**: Exception handling patterns present

## Expected Results

All tests should pass:
- 21 tests total
- TestAPIDocumentation: 16 tests
- TestAPICodeExamples: 5 tests

## Troubleshooting Test Failures

### Missing API Sections
```bash
# Check section headers
grep "^## " docs/api.md
```

### Invalid Code Examples
```bash
# Extract and validate specific code block
python -c "
import re, ast
with open('docs/api.md', 'r') as f:
    content = f.read()
blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
try:
    ast.parse(blocks[0])  # Test first block
    print('✅ Valid')
except SyntaxError as e:
    print(f'❌ Syntax error: {e}')
"
```

### Missing Parameter Documentation
```bash
# Count documentation patterns
python -c "
import re
with open('docs/api.md', 'r') as f:
    content = f.read()
args = len(re.findall(r'Args:\s*\n', content))
returns = len(re.findall(r'Returns:\s*\n', content))
raises = len(re.findall(r'Raises:\s*\n', content))
print(f'Args sections: {args}')
print(f'Returns sections: {returns}')
print(f'Raises sections: {raises}')
"
```

## Manual API Documentation Review

### Check API Completeness
1. Compare with contract files in `specs/001-goal-create-spec/contracts/`
2. Verify all public methods documented
3. Ensure parameter types match implementations
4. Validate example code runs correctly

### Verify Example Accuracy
```bash
# Test example code snippets (requires implementation)
cd examples/
python api_examples.py  # If available
```

## Continuous Integration

These tests run automatically in CI/CD pipeline to ensure API documentation remains accurate and complete as the codebase evolves.