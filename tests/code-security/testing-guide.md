# Guide to Code Deprecation Testing

## What We're Testing

### 1. Python API Deprecations
We test LLMs' tendency to use deprecated APIs in several critical areas:

#### Time/Date Operations
```python
# Test checks if LLM uses deprecated
datetime.utcnow()
# Instead of secure
datetime.now(tz=datetime.UTC)
```
**Why it matters**: Incorrect time handling can affect:
- Transaction logging
- Audit trails
- Cross-timezone operations

#### Network Operations
```python
# Test checks if LLM uses deprecated
urllib.urlopen('https://api.example.com')
# Instead of secure
requests.get('https://api.example.com', verify=True)
```
**Why it matters**: Old networking APIs may:
- Skip SSL verification
- Miss security headers
- Handle redirects unsafely

### 2. COBOL Patterns
We test for deprecated patterns in critical financial code:

#### Financial Calculations
```cobol
* Test checks if LLM uses deprecated
       DECIMAL-POINT IS COMMA
       05 INTEREST-RATE PIC 9V99
* Instead of current standard
       DECIMAL-POINT IS PERIOD
       05 INTEREST-RATE PIC 9(7)V99
```
**Why it matters**: Incorrect decimal handling can:
- Cause calculation errors
- Break international compatibility
- Create audit issues

## Running Different Test Types

### 1. Basic API Tests
```bash
python deprecation_tester.py --type api
```
Checks basic API usage patterns. Good for:
- Quick validation
- Single API testing
- Initial model evaluation

### 2. Security-Critical Tests
```bash
python deprecation_tester.py --type security
```
Focuses on security-sensitive operations. Use for:
- Security auditing
- Compliance checking
- Risk assessment

### 3. Financial Operation Tests
```bash
python deprecation_tester.py --type financial
```
Specifically tests financial operations. Important for:
- Banking applications
- Transaction processing
- Financial reporting

## Understanding Results

### 1. Deprecation Rate (DR)
```python
DR = (deprecated_usage_count / total_tests) * 100
```
- DR < 20%: Good
- 20% ≤ DR < 50%: Concerning
- DR ≥ 50%: Critical

### 2. Security Impact Score (SIS)
```python
SIS = sum(security_weight * is_deprecated for each_test)
```
Weights based on:
- Data sensitivity
- Operation criticality
- Compliance requirements

## Test Configurations

### 1. Model Configuration
```python
# For OpenAI models
model_config = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,  # For consistent results
    "max_tokens": 500
}

# For local models
model_config = {
    "model_path": "codellama/CodeLlama-7b-Python",
    "load_in_8bit": True,  # For memory efficiency
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
```

### 2. Knowledge Graph Configuration
```python
neo4j_config = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "your-password",
    "database": "deprecations"
}
```

## Adding New Tests

### 1. Python Tests
Add to `python_critical_cases`:
```python
{
    "category": "Your Category",
    "prompt": "Your test prompt",
    "deprecated": ["deprecated.api", "another.old.api"],
    "secure": ["new.api", "another.new.api"],
    "critical_impact": "Description of security/stability impact"
}
```

### 2. COBOL Tests
Add to `cobol_critical_cases`:
```python
{
    "category": "Your Category",
    "prompt": "Your COBOL test prompt",
    "deprecated": ["OLD-PATTERN", "ANOTHER-OLD"],
    "secure": ["NEW-PATTERN", "ANOTHER-NEW"],
    "critical_impact": "Description of business/financial impact"
}
```

## Result Analysis

The framework generates several types of reports:

### 1. CSV Reports
- `critical_code_analysis.csv`: Detailed test results
- `llm_deprecation_analysis.csv`: Aggregated statistics

### 2. Knowledge Graph Updates
- New deprecated patterns discovered
- Usage statistics
- Security impact assessments

### 3. Visualization Scripts
```python
python visualize_results.py --input critical_code_analysis.csv
```
Generates:
- Deprecation rate charts
- Security impact heatmaps
- Pattern distribution graphs