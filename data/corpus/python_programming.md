# Python Programming — Study Notes

---

## Variables and Data Types

### What is a Variable?
A variable is a named container that stores a value in memory. In Python, you create a variable simply by assigning a value to it — no type declaration needed.

```python
name = "Alice"       # str
age  = 20            # int
gpa  = 3.8           # float
active = True        # bool
```

### Core Data Types
| Type | Example | Description |
|------|---------|-------------|
| `int` | `42`, `-5` | Whole numbers |
| `float` | `3.14`, `-0.5` | Decimal numbers |
| `str` | `"hello"` | Text (immutable) |
| `bool` | `True`, `False` | Logical values |

### Type Conversion
You can convert between types using built-in functions:
```python
int("42")       # → 42
float("3.14")   # → 3.14
str(100)        # → "100"
bool(0)         # → False  (0, "", [], None are all False)
bool(1)         # → True
```

### Checking Types
```python
x = 3.14
print(type(x))        # <class 'float'>
print(isinstance(x, float))  # True
```

### Key Rules for Variable Names
- Must start with a letter or underscore: `my_var`, `_count`
- Cannot start with a number: ❌ `2name`
- Cannot use reserved words: ❌ `class`, `for`, `if`
- Case-sensitive: `Name` and `name` are different variables

### Integer Division
```python
10 / 3    # → 3.3333  (true division, always float)
10 // 3   # → 3       (floor division, integer result)
10 % 3    # → 1       (modulo: remainder)
2 ** 8    # → 256     (exponentiation)
```

---

## Control Flow

### The if / elif / else Statement
Control flow lets your program make decisions based on conditions.

```python
score = 75

if score >= 90:
    grade = "A"
elif score >= 75:
    grade = "B"
elif score >= 60:
    grade = "C"
else:
    grade = "F"

print(grade)   # B
```

### Comparison Operators
| Operator | Meaning | Example |
|----------|---------|---------|
| `==` | Equal to | `5 == 5` → True |
| `!=` | Not equal | `5 != 3` → True |
| `<` | Less than | `3 < 5` → True |
| `>` | Greater than | `5 > 3` → True |
| `<=` | Less than or equal | `5 <= 5` → True |
| `>=` | Greater than or equal | `6 >= 5` → True |

### Logical Operators
```python
# and: both conditions must be True
x = 7
if x > 5 and x < 10:
    print("Between 5 and 10")   # prints

# or: at least one must be True
if x < 5 or x > 6:
    print("Outside 5-6")        # prints

# not: flips True/False
if not (x == 0):
    print("x is non-zero")      # prints
```

### Nested Conditions
```python
age = 20
has_id = True

if age >= 18:
    if has_id:
        print("Entry allowed")
    else:
        print("Need ID")
else:
    print("Too young")
```

### Common Pitfall
```python
x = 5
if x = 5:     # ❌ SyntaxError — use == not = in conditions
    print("yes")

if x == 5:    # ✅ Correct
    print("yes")
```

---

## Loops

### The for Loop
Use a `for` loop when you know how many times to repeat, or when iterating over a collection.

```python
# range(stop) — 0 to stop-1
for i in range(5):
    print(i)          # prints 0 1 2 3 4

# range(start, stop)
for i in range(1, 6):
    print(i)          # prints 1 2 3 4 5

# range(start, stop, step)
for i in range(0, 10, 2):
    print(i)          # prints 0 2 4 6 8
```

### The while Loop
Use a `while` loop when you repeat until a condition becomes False.

```python
count = 0
while count < 5:
    print(count)
    count += 1        # prints 0 1 2 3 4
```

### break and continue
```python
# break — exit the loop immediately
for i in range(10):
    if i == 5:
        break
    print(i)          # prints 0 1 2 3 4

# continue — skip current iteration, continue loop
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)          # prints 1 3 5 7 9
```

### Iterating Over a List
```python
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# With index using enumerate
for i, fruit in enumerate(fruits):
    print(i, fruit)   # 0 apple, 1 banana, 2 cherry
```

### Common Pattern: Accumulating a Sum
```python
total = 0
for num in range(1, 6):
    total += num
print(total)   # 15  (1+2+3+4+5)
```

---

## Functions

### Defining and Calling a Function
A function is a reusable block of code. Define it once, call it many times.

```python
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")   # Hello, Alice!
greet("Bob")     # Hello, Bob!
```

### Return Values
```python
def add(a, b):
    return a + b

result = add(3, 4)
print(result)    # 7
```

### Default Parameters
```python
def greet(name, message="Hello"):
    print(f"{message}, {name}!")

greet("Alice")              # Hello, Alice!
greet("Bob", "Good morning")# Good morning, Bob!
```

### Variable Scope
Variables defined **inside** a function are **local** — they only exist within that function.
```python
def my_func():
    x = 10        # local variable
    print(x)

my_func()         # 10
print(x)          # ❌ NameError: x is not defined outside
```

Variables defined **outside** all functions are **global**.
```python
total = 0         # global variable

def add_to_total(n):
    global total  # must declare to modify global
    total += n

add_to_total(5)
print(total)      # 5
```

### *args — Variable Number of Arguments
```python
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3))      # 6
print(sum_all(1, 2, 3, 4, 5))# 15
```

### Functions are First-Class Objects
```python
def square(x):
    return x ** 2

apply = square         # assign function to variable
print(apply(5))        # 25
```

---

## OOP Basics

### What is OOP?
Object-Oriented Programming (OOP) organises code around **objects** — bundles of data (attributes) and behaviour (methods). A **class** is the blueprint; an **object** (or instance) is a specific copy created from that blueprint.

```python
# Blueprint (class)
class Dog:
    def __init__(self, name, breed):
        self.name  = name    # instance attribute
        self.breed = breed

    def bark(self):
        print(f"{self.name} says: Woof!")

# Creating objects (instances)
dog1 = Dog("Rex", "Labrador")
dog2 = Dog("Bella", "Poodle")

dog1.bark()   # Rex says: Woof!
dog2.bark()   # Bella says: Woof!
```

### `__init__` — The Constructor
`__init__` is automatically called when you create a new object. It initialises the object's attributes.
```python
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

c = Circle(5)
print(c.area())   # 78.54
```

### `self`
`self` refers to the **current instance**. Every instance method must have `self` as its first parameter.
```python
class Counter:
    def __init__(self):
        self.count = 0       # each instance has its own count

    def increment(self):
        self.count += 1

    def get(self):
        return self.count

c1 = Counter()
c2 = Counter()
c1.increment()
c1.increment()
print(c1.get())   # 2
print(c2.get())   # 0  (separate instance)
```

### Inheritance
A child class **inherits** all methods and attributes from its parent class, and can add or override them.
```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print(f"{self.name} makes a sound.")

class Cat(Animal):           # Cat inherits from Animal
    def speak(self):         # override parent method
        print(f"{self.name} says: Meow!")

class Dog(Animal):
    def speak(self):
        print(f"{self.name} says: Woof!")

animals = [Cat("Whiskers"), Dog("Rex"), Cat("Luna")]
for a in animals:
    a.speak()
# Whiskers says: Meow!
# Rex says: Woof!
# Luna says: Meow!
```

### `super()` — Calling the Parent Constructor
```python
class Vehicle:
    def __init__(self, brand):
        self.brand = brand

class Car(Vehicle):
    def __init__(self, brand, model):
        super().__init__(brand)   # call Vehicle.__init__
        self.model = model

car = Car("Toyota", "Corolla")
print(car.brand, car.model)       # Toyota Corolla
```

### Key OOP Terms
| Term | Meaning |
|------|---------|
| **Class** | Blueprint for creating objects |
| **Object/Instance** | A specific copy of a class |
| **Attribute** | Data stored on an object (`self.name`) |
| **Method** | Function defined inside a class |
| **Inheritance** | Child class reuses parent class code |
| **`__init__`** | Constructor — runs when object is created |
| **`self`** | Reference to the current instance |
