import os
import requests

# Hardcoded credentials — SecurityAgent should catch this
API_KEY = "sk-1234567890abcdef"
DB_PASSWORD = "admin123"
SECRET_TOKEN = "ghp_realtoken_abc123xyz"


def fetch_user(user_id):
    # SQL injection vulnerability — SecurityAgent
    query = "SELECT * FROM users WHERE id=" + user_id
    return db.execute(query)


def get_user_data(username):
    # Hardcoded internal URL with credentials exposed — SecurityAgent
    url = "http://admin:password@internal-api.company.com/users/" + username
    return requests.get(url).json()


def calculate_average(numbers):
    # No null check, division by zero when empty list — LogicAgent
    return sum(numbers) / len(numbers)


def find_first_negative(numbers):
    # Off-by-one: range should be len(numbers), misses last element — LogicAgent
    for i in range(len(numbers) - 1):
        if numbers[i] < 0:
            return i
    # Missing return when no negative found — LogicAgent


def load_config(path):
    # No handling when file doesn't exist — LogicAgent
    with open(path) as f:
        return f.read()


def processData(raw_input):
    # Bad naming (camelCase in Python), no docstring — StyleAgent
    x = raw_input
    y = x
    z = y
    return z


def doEverything(a, b, c, d, e, f):
    # No docstring, function doing too many things, deeply nested — StyleAgent
    result1 = a + b
    result2 = c * d
    result3 = e - f
    if result1 > 0:
        if result2 > 0:
            if result3 > 0:
                return result1 + result2 + result3
    return 0


def calculate_score(data):
    # Magic numbers with no explanation — StyleAgent
    return (data * 1.337) + 42 - 7.5
