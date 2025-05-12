# Integer
age = 25
print(type(age))  # <class 'int'>

# Float
price = 19.99
print(type(price))  # <class 'float'>

# String
name = "Alice"
print(type(name))  # <class 'str'>

# Boolean
is_student = True
print(type(is_student))  # <class 'bool'>

# NoneType
result = None
print(type(result))  # <class 'NoneType'>


# Arithmetic operations
a = 10
b = 3

print(a + b)  # 13 (Addition)
print(a - b)  # 7 (Subtraction)
print(a * b)  # 30 (Multiplication)
print(a / b)  # 3.333... (Division)
print(a // b) # 3 (Floor division)
print(a % b)  # 1 (Modulus)
print(a ** b) # 1000 (Exponentiation)

# String operations
first_name = "John"
last_name = "Doe"

full_name = first_name + " " + last_name  # Concatenation
print(full_name)  # "John Doe"

print(first_name * 3)  # "JohnJohnJohn"
print("oh" in first_name)  # True (Membership check)

# Basic output
print("Hello, World!")

# Formatted output
name = "Alice"
age = 25
print(f"My name is {name} and I'm {age} years old.")  # f-string (Python 3.6+)
print("My name is {} and I'm {} years old.".format(name, age))  # format method

# User input
user_name = input("What's your name? ")
print(f"Hello, {user_name}!")

# Note: input() always returns a string
age = input("How old are you? ")
print(type(age))  # <class 'str'>
# To convert to integer:
age = int(age)

temperature = 30

if temperature > 30:
    print("It's hot outside!")
elif 20 <= temperature <= 30:
    print("The weather is nice.")
else:
    print("It's cold outside!")
    
# Iterate through a range
for i in range(5):  # 0 to 4
    print(i)

# Iterate through a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# With index
for index, fruit in enumerate(fruits):
    print(f"Index {index}: {fruit}")
    
count = 0
while count < 5:
    print(count)
    count += 1  # Important: Don't forget to increment!
    
# Create a list
numbers = [1, 2, 3, 4, 5]
fruits = ["apple", "banana", "cherry"]

# Access elements
print(numbers[0])  # 1 (first element)
print(numbers[-1])  # 5 (last element)

# Modify elements
numbers[0] = 10

# List operations
numbers.append(6)  # Add to end
numbers.insert(1, 1.5)  # Insert at position
numbers.remove(2)  # Remove first occurrence
popped = numbers.pop()  # Remove and return last item

# List slicing
print(numbers[1:3])  # Elements from index 1 to 2
print(numbers[:3])   # First 3 elements
print(numbers[2:])   # From index 2 to end
print(numbers[::-1]) # Reverse the list

# Create a tuple (immutable)
coordinates = (10, 20)

# Access elements
print(coordinates[0])  # 10

# Unpacking
x, y = coordinates
print(x, y)  # 10 20

# Create a dictionary
person = {
    "name": "John",
    "age": 30,
    "city": "New York"
}
# Access values
print(person["name"])  # John


# Create a set (unique elements)
unique_numbers = {1, 2, 3, 3, 4}
print(unique_numbers)  # {1, 2, 3, 4}

# Set operations
a = {1, 2, 3}
b = {3, 4, 5}

print(a | b)  # Union: {1, 2, 3, 4, 5}
print(a & b)  # Intersection: {3}
print(a - b)  # Difference: {1, 2}

