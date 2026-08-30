# Python Programming — Reference Corpus

## Variables and Data Types
Python variables are created by assignment: `x = 5`. Python is dynamically typed — the type is inferred at runtime. Core types: `int` (whole numbers), `float` (decimals), `str` (text, immutable), `bool` (`True`/`False`). Type conversion: `int("3")`, `float(3)`, `str(42)`. The `type()` function returns an object's type.

## Operators and Expressions
Arithmetic: `+`, `-`, `*`, `/` (true division → float), `//` (floor division), `%` (modulo), `**` (exponentiation). Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=` return bool. Logical: `and`, `or`, `not`. Operator precedence follows standard BODMAS; use parentheses to override.

## Strings
Strings are immutable sequences of Unicode characters. Indexing: `s[0]` (first), `s[-1]` (last). Slicing: `s[start:stop:step]`. Methods: `.upper()`, `.lower()`, `.strip()`, `.split()`, `.join()`, `.replace()`, `.find()`. f-strings: `f"Hello {name}"`. String concatenation with `+`, repetition with `*`.

## Control Flow
`if condition:` executes a block when condition is True. `elif` checks additional conditions only if prior ones were False. `else` runs when none match. Indentation defines scope — 4 spaces is PEP8 standard. Ternary: `x if condition else y`.

## Loops
`for item in iterable:` iterates over sequences, ranges, or any iterable. `while condition:` repeats until condition is False. `break` exits the loop immediately. `continue` skips to the next iteration. `else` clause on loops executes when loop completes normally (no `break`). `range(start, stop, step)` generates integer sequences.

## Lists and Tuples
Lists are mutable ordered sequences: `[1, 2, 3]`. Methods: `.append()`, `.extend()`, `.insert()`, `.remove()`, `.pop()`, `.sort()`, `.reverse()`, `len()`. Tuples are immutable: `(1, 2, 3)`. Use tuples for fixed data, lists for mutable collections. List comprehension: `[x**2 for x in range(10) if x % 2 == 0]`.

## Dictionaries and Sets
Dictionaries store key-value pairs: `{"key": value}`. Keys must be hashable (str, int, tuple). Methods: `.keys()`, `.values()`, `.items()`, `.get(key, default)`, `.update()`, `.pop()`. Sets store unique elements: `{1, 2, 3}`. Set operations: union `|`, intersection `&`, difference `-`. `frozenset` is immutable.

## Functions
Define with `def name(params):`. Return values with `return`. Default parameters: `def f(x=5)`. `*args` collects extra positional args as tuple; `**kwargs` collects keyword args as dict. Functions are first-class objects — they can be passed as arguments, returned, stored in variables. Lambda: `lambda x: x*2`.

## Scope and Recursion
LEGB rule: Local → Enclosing → Global → Built-in. `global` keyword accesses module-level variable inside a function. `nonlocal` accesses enclosing function's variable. Recursion: a function calling itself. Every recursive function needs a base case to terminate. Python default recursion limit is 1000 (`sys.setrecursionlimit()`).

## Exceptions
`try:` block contains risky code. `except ExceptionType:` handles specific errors. `except (TypeError, ValueError):` catches multiple. `else:` runs if no exception. `finally:` always runs (cleanup). `raise` re-raises or creates exceptions. Common exceptions: `ValueError`, `TypeError`, `IndexError`, `KeyError`, `FileNotFoundError`, `ZeroDivisionError`.

## File I/O
`open(filename, mode)` — modes: `'r'` (read), `'w'` (write, overwrites), `'a'` (append), `'rb'`/`'wb'` (binary). Always use context managers: `with open('file.txt', 'r') as f:` — guarantees file is closed even if exception occurs. `f.read()` reads entire file; `f.readline()` one line; `f.readlines()` list of lines; `f.write(str)` writes.

## OOP Basics
`class ClassName:` defines a class. `__init__(self, ...)` constructor. `self` refers to the instance. Instance attributes: `self.name = value`. Class attributes: defined at class level. Inheritance: `class Child(Parent):`. `super().__init__()` calls parent constructor. `__str__` defines string representation. Encapsulation via naming convention: `_private`, `__mangled`.
