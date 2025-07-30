# Tests

## Description: 
If you write functions that play a major role in the physics part of this project, you must write a test for them.

## How to Run Tests:
In the *proect root* directory (i.e., the one containing the `tests/` folder), run:

```python
python tests/[name_of_test_file].py
```

Parenthetical: This works because every test file features the following lines of code at the bottom:
```python
if __name__ == "__main__":
    unittest.main()
```
