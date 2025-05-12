def greet(name):
    """This function greets the person passed in as parameter"""
    print(f"Hello, {name}!")

greet("Alice")  # Output: Hello, Alice!

def add_numbers(a, b):
    """Returns the sum of two numbers"""
    return a + b

result = add_numbers(5, 3)
print(result)  # Output: 8

def power(base, exponent=2):
    """Returns base raised to the power of exponent (default is 2)"""
    return base ** exponent

print(power(3))     # Output: 9 (3^2)
print(power(3, 3))  # Output: 27 (3^3)

def average(*numbers):
    """Calculates average of any number of values"""
    return sum(numbers) / len(numbers)

print(average(1, 2, 3))      # Output: 2.0
print(average(10, 20, 30, 40))  # Output: 25.0

square = lambda x: x ** 2
print(square(5))  # Output: 25

# Often used with map(), filter(), etc.
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # Output: [1, 4, 9, 16]

class Dog:
    """A simple Dog class"""
    
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    # Initializer (constructor)
    def __init__(self, name, age):
        # Instance attributes
        self.name = name
        self.age = age
    
    # Instance method
    def description(self):
        return f"{self.name} is {self.age} years old"
    
    # Another instance method
    def speak(self, sound):
        return f"{self.name} says {sound}"

# Create instances
dog1 = Dog("Buddy", 5)
dog2 = Dog("Milo", 3)

# Access attributes and methods
print(dog1.name)  # Output: Buddy
print(dog2.description())  # Output: Milo is 3 years old
print(dog1.speak("Woof!"))  # Output: Buddy says Woof!
print(dog1.species)  # Output: Canis familiaris

class Animal:
    """Base class for all animals"""
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement this method")

class Cat(Animal):
    """Cat class inherits from Animal"""
    def speak(self):
        return f"{self.name} says Meow!"

class Cow(Animal):
    """Cow class inherits from Animal"""
    def speak(self):
        return f"{self.name} says Moo!"

# Create instances
cat = Cat("Whiskers")
cow = Cow("Bessie")

print(cat.speak())  # Output: Whiskers says Meow!
print(cow.speak())  # Output: Bessie says Moo!

class MyClass:
    class_attribute = "I'm a class attribute"
    
    def __init__(self, value):
        self.instance_attribute = value
    
    @classmethod
    def class_method(cls):
        """Can access class attributes but not instance attributes"""
        return f"Class method called. Class attribute: {cls.class_attribute}"
    
    @staticmethod
    def static_method():
        """Can't access class or instance attributes - just a utility function"""
        return "Static method called"

# Using class method
print(MyClass.class_method())  # Output: Class method called. Class attribute: I'm a class attribute

# Using static method
print(MyClass.static_method())  # Output: Static method called

class Circle:
    def __init__(self, radius):
        self._radius = radius  # Protected attribute
    
    @property
    def radius(self):
        """Getter for radius"""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Setter for radius with validation"""
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value
    
    @property
    def area(self):
        """Calculated property (read-only)"""
        return 3.14159 * self._radius ** 2

circle = Circle(5)
print(circle.radius)  # Output: 5 (using getter)
print(circle.area)    # Output: 78.53975

circle.radius = 7     # Using setter
print(circle.area)    # Output: 153.93791

# circle.radius = -1  # Raises ValueError: Radius must be positive

class Vector:
    """A simple vector class demonstrating magic methods"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Vector addition"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Vector subtraction"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Scalar multiplication"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __repr__(self):
        """Official string representation"""
        return f"Vector({self.x}, {self.y})"
    
    def __len__(self):
        """Length of vector (number of dimensions)"""
        return 2
    
    def __eq__(self, other):
        """Equality comparison"""
        return self.x == other.x and self.y == other.y

# Using the Vector class
v1 = Vector(2, 4)
v2 = Vector(1, 3)

print(v1 + v2)  # Output: Vector(3, 7) (uses __add__)
print(v1 - v2)  # Output: Vector(1, 1) (uses __sub__)
print(v1 * 3)   # Output: Vector(6, 12) (uses __mul__)
print(len(v1))  # Output: 2 (uses __len__)
print(v1 == Vector(2, 4))  # Output: True (uses __eq__)