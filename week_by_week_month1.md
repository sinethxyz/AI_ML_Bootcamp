# Month 1: Detailed Day-by-Day Schedule
## Foundation Phase - Python, Math, Data

**Intensity:** 40 hours/week (8 hours/day, 5 days + weekend projects)
**Goal:** Solid foundation in Python, essential math, and data manipulation

---

## WEEK 1: Python Fundamentals

### Monday - Day 1: Variables, Data Types, Basic Operations

**Morning (4 hours): Learn**
```python
# Hour 1: Variables and Basic Types
age = 25
name = "Data Scientist"
height = 5.9
is_learning = True

# Hour 2: Operations
x = 10
y = 3
print(x + y, x - y, x * y, x / y, x // y, x % y, x ** y)

# Hour 3: Strings
text = "Machine Learning"
print(text.lower(), text.upper(), text.split())
print(f"Learning {text}")

# Hour 4: Lists Basics
numbers = [1, 2, 3, 4, 5]
numbers.append(6)
numbers.pop()
print(numbers[0], numbers[-1], numbers[1:3])
```

**Afternoon (4 hours): Build**
```python
# PROJECT: Calculator with history
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a, b):
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def show_history(self):
        for item in self.history:
            print(item)

# Use it
calc = Calculator()
calc.add(5, 3)
calc.subtract(10, 4)
calc.show_history()
```

**Daily Goals:**
- [ ] Write 100+ lines of code
- [ ] Build working calculator
- [ ] Push to GitHub
- [ ] Tweet progress

---

### Tuesday - Day 2: Control Flow & Functions

**Morning (4 hours): Learn**
```python
# Hour 1-2: If/Else
def classify_number(n):
    if n > 0:
        return "positive"
    elif n < 0:
        return "negative"
    else:
        return "zero"

# For loops
for i in range(10):
    print(i ** 2)

# While loops
i = 0
while i < 10:
    print(i)
    i += 1

# Hour 3-4: Functions
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Lambda functions
square = lambda x: x ** 2
numbers = [1, 2, 3, 4, 5]
squared = list(map(square, numbers))
```

**Afternoon (4 hours): Build**
```python
# PROJECT: Number Analysis Tool
def analyze_numbers(numbers):
    """Complete analysis of a list of numbers"""
    results = {
        'count': len(numbers),
        'sum': sum(numbers),
        'mean': sum(numbers) / len(numbers),
        'min': min(numbers),
        'max': max(numbers),
        'range': max(numbers) - min(numbers),
        'positive': len([n for n in numbers if n > 0]),
        'negative': len([n for n in numbers if n < 0]),
        'even': len([n for n in numbers if n % 2 == 0]),
        'odd': len([n for n in numbers if n % 2 != 0])
    }
    return results

def prime_numbers(limit):
    """Find all prime numbers up to limit"""
    primes = []
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

# Test it
nums = [1, -2, 3, -4, 5, 6, 7, 8, 9, 10]
print(analyze_numbers(nums))
print(prime_numbers(100))
```

**Daily Goals:**
- [ ] Master control flow
- [ ] Write 10+ functions
- [ ] Build analysis tool
- [ ] Document all functions

---

### Wednesday - Day 3: Data Structures (Dicts, Sets, Tuples)

**Morning (4 hours): Learn**
```python
# Hour 1-2: Dictionaries
person = {
    'name': 'Alice',
    'age': 25,
    'skills': ['Python', 'ML', 'Stats']
}

# Dictionary operations
person['city'] = 'NYC'
person.get('salary', 0)
person.keys()
person.values()
person.items()

# Nested dictionaries
data = {
    'users': {
        'alice': {'age': 25, 'score': 95},
        'bob': {'age': 30, 'score': 87}
    }
}

# Hour 3: Sets
skills = {'Python', 'SQL', 'ML'}
more_skills = {'ML', 'DL', 'NLP'}

# Set operations
skills.union(more_skills)
skills.intersection(more_skills)
skills.difference(more_skills)

# Hour 4: Tuples & Named Tuples
coordinates = (10, 20)
x, y = coordinates

from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)
```

**Afternoon (4 hours): Build**
```python
# PROJECT: Student Grade Management System
class GradeBook:
    def __init__(self):
        self.students = {}
    
    def add_student(self, name):
        if name not in self.students:
            self.students[name] = []
    
    def add_grade(self, name, subject, grade):
        if name in self.students:
            self.students[name].append({
                'subject': subject,
                'grade': grade
            })
    
    def get_average(self, name):
        if name in self.students:
            grades = [g['grade'] for g in self.students[name]]
            return sum(grades) / len(grades) if grades else 0
    
    def get_top_students(self, n=3):
        averages = {
            name: self.get_average(name) 
            for name in self.students
        }
        sorted_students = sorted(
            averages.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_students[:n]
    
    def subject_statistics(self, subject):
        grades = []
        for student_grades in self.students.values():
            for g in student_grades:
                if g['subject'] == subject:
                    grades.append(g['grade'])
        
        return {
            'count': len(grades),
            'average': sum(grades) / len(grades),
            'max': max(grades),
            'min': min(grades)
        }

# Use it
gb = GradeBook()
gb.add_student('Alice')
gb.add_grade('Alice', 'Math', 95)
gb.add_grade('Alice', 'Science', 88)
print(gb.get_average('Alice'))
print(gb.subject_statistics('Math'))
```

**Daily Goals:**
- [ ] Master all data structures
- [ ] Build gradebook system
- [ ] Handle nested data
- [ ] Write unit tests

---

### Thursday - Day 4: List Comprehensions & File I/O

**Morning (4 hours): Learn**
```python
# Hour 1-2: List Comprehensions
# Basic
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Multiple conditions
filtered = [x for x in range(100) if x % 2 == 0 if x % 3 == 0]

# Nested
matrix = [[i*j for j in range(5)] for i in range(5)]

# Dict comprehension
word_lengths = {word: len(word) for word in ['Python', 'ML', 'AI']}

# Set comprehension
unique_lengths = {len(word) for word in ['hello', 'world', 'hi']}

# Hour 3-4: File I/O
# Read file
with open('data.txt', 'r') as f:
    content = f.read()
    lines = f.readlines()

# Write file
with open('output.txt', 'w') as f:
    f.write('Hello, World!\n')
    f.writelines(['Line 1\n', 'Line 2\n'])

# CSV handling
import csv

# Write CSV
with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age', 'Score'])
    writer.writerow(['Alice', 25, 95])

# Read CSV
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# JSON handling
import json

data = {'name': 'Alice', 'scores': [95, 88, 92]}

# Write JSON
with open('data.json', 'w') as f:
    json.dump(data, f, indent=2)

# Read JSON
with open('data.json', 'r') as f:
    loaded = json.load(f)
```

**Afternoon (4 hours): Build**
```python
# PROJECT: Data Processing Pipeline
import csv
import json
from pathlib import Path

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def load_csv(self, filename):
        """Load data from CSV file"""
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            self.data = list(reader)
        return self
    
    def filter(self, condition):
        """Filter data based on condition function"""
        self.data = [row for row in self.data if condition(row)]
        return self
    
    def transform(self, field, function):
        """Transform a specific field"""
        for row in self.data:
            row[field] = function(row[field])
        return self
    
    def aggregate(self, field):
        """Calculate statistics for a field"""
        values = [float(row[field]) for row in self.data]
        return {
            'count': len(values),
            'sum': sum(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }
    
    def save_csv(self, filename):
        """Save processed data to CSV"""
        if self.data:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
                writer.writeheader()
                writer.writerows(self.data)
        return self
    
    def save_json(self, filename):
        """Save processed data to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)
        return self

# Use it
processor = DataProcessor()
processor.load_csv('sales.csv') \
    .filter(lambda row: int(row['quantity']) > 10) \
    .transform('price', lambda x: float(x) * 1.1) \
    .save_csv('processed_sales.csv')

print(processor.aggregate('quantity'))
```

**Daily Goals:**
- [ ] Master comprehensions
- [ ] Handle files confidently
- [ ] Build data pipeline
- [ ] Process real CSV data

---

### Friday - Day 5: Error Handling & Debugging

**Morning (4 hours): Learn**
```python
# Hour 1-2: Try/Except
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero!")
        return None
    except TypeError:
        print("Invalid types!")
        return None
    else:
        print(f"Success: {result}")
        return result
    finally:
        print("Division operation completed")

# Custom exceptions
class ValidationError(Exception):
    pass

def validate_age(age):
    if age < 0:
        raise ValidationError("Age cannot be negative!")
    if age > 150:
        raise ValidationError("Age too high!")
    return True

# Hour 3: Debugging techniques
import logging

logging.basicConfig(level=logging.DEBUG)

def complex_function(data):
    logging.debug(f"Input data: {data}")
    
    # Process
    result = process(data)
    logging.info(f"Processed: {result}")
    
    return result

# Hour 4: Assert statements
def calculate_percentage(part, total):
    assert total > 0, "Total must be positive"
    assert part <= total, "Part cannot exceed total"
    return (part / total) * 100
```

**Afternoon (4 hours): Build**
```python
# PROJECT: Robust Data Validator
import logging
from typing import List, Dict, Any

class DataValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        logging.basicConfig(level=logging.INFO)
    
    def validate_required_fields(self, data: Dict, required: List[str]) -> bool:
        """Check if all required fields are present"""
        missing = [field for field in required if field not in data]
        if missing:
            error = f"Missing required fields: {missing}"
            self.errors.append(error)
            logging.error(error)
            return False
        return True
    
    def validate_type(self, value: Any, expected_type: type, field_name: str) -> bool:
        """Validate field type"""
        if not isinstance(value, expected_type):
            error = f"{field_name}: expected {expected_type}, got {type(value)}"
            self.errors.append(error)
            logging.error(error)
            return False
        return True
    
    def validate_range(self, value: float, min_val: float, max_val: float, field_name: str) -> bool:
        """Validate numeric range"""
        if not (min_val <= value <= max_val):
            error = f"{field_name}: {value} not in range [{min_val}, {max_val}]"
            self.errors.append(error)
            logging.error(error)
            return False
        return True
    
    def validate_email(self, email: str) -> bool:
        """Basic email validation"""
        if '@' not in email or '.' not in email:
            error = f"Invalid email format: {email}"
            self.errors.append(error)
            logging.error(error)
            return False
        return True
    
    def validate_dataset(self, data: List[Dict], schema: Dict) -> bool:
        """Validate entire dataset against schema"""
        self.errors = []
        self.warnings = []
        
        for i, row in enumerate(data):
            # Check required fields
            if not self.validate_required_fields(row, schema.get('required', [])):
                continue
            
            # Check field types and constraints
            for field, constraints in schema.get('fields', {}).items():
                if field not in row:
                    continue
                
                value = row[field]
                
                # Type validation
                if 'type' in constraints:
                    if not self.validate_type(value, constraints['type'], f"Row {i}.{field}"):
                        continue
                
                # Range validation
                if 'min' in constraints and 'max' in constraints:
                    try:
                        self.validate_range(
                            float(value),
                            constraints['min'],
                            constraints['max'],
                            f"Row {i}.{field}"
                        )
                    except ValueError:
                        error = f"Row {i}.{field}: cannot convert to number"
                        self.errors.append(error)
        
        return len(self.errors) == 0
    
    def get_report(self) -> Dict:
        """Get validation report"""
        return {
            'valid': len(self.errors) == 0,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings
        }

# Example usage
validator = DataValidator()

schema = {
    'required': ['name', 'age', 'email'],
    'fields': {
        'name': {'type': str},
        'age': {'type': int, 'min': 0, 'max': 150},
        'email': {'type': str}
    }
}

data = [
    {'name': 'Alice', 'age': 25, 'email': 'alice@email.com'},
    {'name': 'Bob', 'age': -5, 'email': 'invalid'},  # Invalid
    {'name': 'Charlie', 'age': 30}  # Missing email
]

validator.validate_dataset(data, schema)
print(validator.get_report())
```

**Daily Goals:**
- [ ] Handle all error types
- [ ] Build robust validator
- [ ] Add logging
- [ ] Write comprehensive tests

---

### Weekend Project: Mini ETL System

**Saturday (6 hours): Build**
```python
# Complete ETL (Extract, Transform, Load) System

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Callable

class ETLPipeline:
    """Extract, Transform, Load pipeline"""
    
    def __init__(self, name: str):
        self.name = name
        self.data = []
        self.transformations = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(name)
    
    # EXTRACT
    def extract_csv(self, filename: str) -> 'ETLPipeline':
        """Extract data from CSV"""
        self.logger.info(f"Extracting from {filename}")
        try:
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                self.data = list(reader)
            self.logger.info(f"Extracted {len(self.data)} rows")
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            raise
        return self
    
    def extract_json(self, filename: str) -> 'ETLPipeline':
        """Extract data from JSON"""
        self.logger.info(f"Extracting from {filename}")
        try:
            with open(filename, 'r') as f:
                self.data = json.load(f)
            self.logger.info(f"Extracted {len(self.data)} records")
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            raise
        return self
    
    # TRANSFORM
    def add_transformation(self, func: Callable) -> 'ETLPipeline':
        """Add a transformation function"""
        self.transformations.append(func)
        return self
    
    def filter_rows(self, condition: Callable) -> 'ETLPipeline':
        """Filter rows based on condition"""
        def transform(data):
            return [row for row in data if condition(row)]
        return self.add_transformation(transform)
    
    def map_field(self, field: str, func: Callable) -> 'ETLPipeline':
        """Transform a specific field"""
        def transform(data):
            for row in data:
                if field in row:
                    row[field] = func(row[field])
            return data
        return self.add_transformation(transform)
    
    def add_field(self, field: str, func: Callable) -> 'ETLPipeline':
        """Add a new field"""
        def transform(data):
            for row in data:
                row[field] = func(row)
            return data
        return self.add_transformation(transform)
    
    def remove_field(self, field: str) -> 'ETLPipeline':
        """Remove a field"""
        def transform(data):
            for row in data:
                row.pop(field, None)
            return data
        return self.add_transformation(transform)
    
    def execute_transformations(self) -> 'ETLPipeline':
        """Execute all transformations"""
        self.logger.info(f"Executing {len(self.transformations)} transformations")
        for i, transform in enumerate(self.transformations):
            self.logger.debug(f"Applying transformation {i+1}")
            self.data = transform(self.data)
        self.logger.info(f"Transformations complete: {len(self.data)} rows remaining")
        return self
    
    # LOAD
    def load_csv(self, filename: str) -> 'ETLPipeline':
        """Load data to CSV"""
        self.logger.info(f"Loading to {filename}")
        if self.data:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
                writer.writeheader()
                writer.writerows(self.data)
            self.logger.info(f"Loaded {len(self.data)} rows to CSV")
        return self
    
    def load_json(self, filename: str) -> 'ETLPipeline':
        """Load data to JSON"""
        self.logger.info(f"Loading to {filename}")
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)
        self.logger.info(f"Loaded {len(self.data)} records to JSON")
        return self
    
    # UTILITY
    def get_stats(self) -> Dict:
        """Get statistics about current data"""
        if not self.data:
            return {}
        
        return {
            'row_count': len(self.data),
            'fields': list(self.data[0].keys()) if self.data else [],
            'field_count': len(self.data[0]) if self.data else 0
        }

# Example usage
pipeline = ETLPipeline("sales_processor")

pipeline.extract_csv('raw_sales.csv') \
    .filter_rows(lambda row: float(row['amount']) > 100) \
    .map_field('amount', lambda x: float(x) * 1.1) \
    .map_field('date', lambda x: x.split()[0]) \
    .add_field('category', lambda row: 'high' if float(row['amount']) > 500 else 'low') \
    .remove_field('internal_id') \
    .execute_transformations() \
    .load_csv('processed_sales.csv') \
    .load_json('processed_sales.json')

print(pipeline.get_stats())
```

**Sunday (4 hours): Document & Polish**
- [ ] Write README for ETL system
- [ ] Add docstrings
- [ ] Write unit tests
- [ ] Create example usage
- [ ] Push to GitHub

---

## Week 1 Success Criteria

**By end of Week 1, you should:**
- [ ] Write 500+ lines of Python
- [ ] Build 8 working projects
- [ ] Comfortable with all basic syntax
- [ ] Handle files confidently
- [ ] GitHub repo with all projects
- [ ] First blog post written

**Next Week:** NumPy, Pandas, and Mathematics fundamentals

---

## Tips for Success

**ADHD Optimization:**
- Set timer for 50-minute work sessions
- Physical movement between sessions
- Tweet every completion
- Visible progress tracker

**Autism Optimization:**
- Follow schedule exactly
- Check off each completed task
- Document everything
- Create pattern library

**Pattern Recognition:**
- Notice recurring patterns
- Create mental models
- Connect to what you know
- Build frameworks

**Speed Tips:**
- Type code, don't copy-paste
- Make mistakes and debug
- Build before it's perfect
- Ship daily