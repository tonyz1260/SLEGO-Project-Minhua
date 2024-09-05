import inspect
import importlib.util
import ast
from typing import Callable
from config import *
import func

from openai import OpenAI

class ValidationResult:
    def __init__(self, function_name: str, validation_result: bool, message: str, issue_type: str):
        self.function_name = function_name
        self.validation_result = validation_result
        self.message = message
        self.issue_type = issue_type
        if self.issue_type not in ["ERROR", "WARNING"]:
            raise ValueError("Invalid type value. Type must be 'ERROR' or 'WARNING'.")

    def __str__(self):
        return f'{self.issue_type}\nFunction: {self.function_name}\nValidation Result: {self.validation_result}\nMessage: {self.message}\n\n'
    
    def get_result(self) -> bool:
        return self.validation_result
    
class ValidationModel:
    def __init__(self, function: Callable, priority: int, description: str, issue_type: str):
        self.function = function
        self.priority = priority
        self.description = description
        self.issue_type = issue_type
        if self.issue_type not in ["ERROR", "WARNING"]:
            raise ValueError("Invalid type value. Type must be 'ERROR' or 'WARNING'.")

# Warnings -------------------------------------------------------------------------------
# Check for function relevance
def check_relevance(name, obj):
    client = OpenAI(api_key=OPENAI_API_KEY2)

    prompt = "You are now a software engineer and your task is to assess the relevance of the functions that are provided to you. \
    You are given a function that is written in Python and you need to determine whether the function is relevant to the existing functions (docstrings of existing functions will be provided) and/or related to the finance domain. \
    In addition, you should also determine whether the function is meant to have the input and output parameters. \
    What that means is based on the actual function context, it may be a function that does webscraping and does have the output path but not the input path. \
    Or it may be a function that does not have the output path but does have the input path. \
    So your final return response should include the result and must strictly follow this format 'Relevance: [true/false]; Parameter: [input/output/both]' \
    You must not add any additional text or characters to the response. From here, I will provide you with the docstrings of existing functions that you need to assess."

    # iterate through the existing functions in func.py and provide the docstrings for the prompt
    for name, obj in inspect.getmembers(func):
        if inspect.isfunction(obj) and obj.__module__ == func.__name__:
            prompt += f"\n\nFunction: {name}\nDocstring: {inspect.getdoc(obj)}\n"

    # append the function (source_code) that needs to be assessed
    prompt += "Here is the function that you need to assess"
    prompt += f"\n\nFunction: {name}\nSource Code: {inspect.getsource(obj)}\n"


    # Create a chat completion request
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
    )

    # Print the response from the model
    print(response.choices[0].message.content)

def check_parameters(name, obj):
    sig = inspect.signature(obj)
    parameters = sig.parameters

    # At the moment, assuming all functions should have at least 2 parameters
    # TODO: Check whether the function is meant to have more than 2 parameters, it may only need 1 parameter
    if len(parameters) < 2:
        return ValidationResult(name, False, 'Function does not have enough parameters', 'WARNING')
    
    return ValidationResult(name, True, 'Function has enough parameters', 'WARNING')


def check_input_output_stream(name, obj):
    sig = inspect.signature(obj)
    parameters = sig.parameters

    has_input_stream = False
    has_output_stream = False
    for param in parameters.values():
        if param.annotation == str and param.default != inspect.Parameter.default:
            if 'input' in param.name:
                has_input_stream = True
            if 'output' in param.name:
                has_output_stream = True

    if not has_input_stream:
        return ValidationResult(name, False, 'Function does not have input stream or not configured correctly','WARNING')
    if not has_output_stream:
        return ValidationResult(name, False, 'Function does not have output stream or not configured correctly','WARNING')
    
    return ValidationResult(name, True, 'Function has input and output stream', 'WARNING')

# Errors -------------------------------------------------------------------------------
def check_docstring(name, obj):
    if inspect.getdoc(obj) is None:
        return ValidationResult(name, False, 'Function does not have a docstring', 'ERROR')
    
    # TODO: Check whether the inspect.getdoc(obj) matches the function's actual behavior
    return ValidationResult(name, True, 'Function docString is valid', 'ERROR')

def check_annotations_and_default_values(name, obj):
    sig = inspect.signature(obj)
    parameters = sig.parameters

    for param in parameters.values():
        if param.annotation == inspect.Parameter.empty:
            return ValidationResult(name, False, f'Parameter {param.name} does not have annotation', 'ERROR')
        if param.default == inspect.Parameter.empty:
            return ValidationResult(name, False, f'Parameter {param.name} does not have default value', 'ERROR')
        
    return ValidationResult(name, True, 'Function has annotations and default values', 'ERROR')

def check_syntax_error(name, obj):
    try:
        ast.parse(inspect.getsource(obj))
    except SyntaxError as e:
        return ValidationResult(name, False, f'Syntax Error on line {e.lineno} character {e.offset}: {e.msg}\nCode: {e.text}', 'ERROR')
    
    return ValidationResult(name, True, 'Function has no syntax error', 'ERROR')



# Create a sorted list of validation models
validation_rules = [
    ValidationModel(check_syntax_error, 1, 'Function does not contain syntax error', 'ERROR'),
    ValidationModel(check_docstring, 2, 'Function contains docString', 'ERROR'),
    ValidationModel(check_annotations_and_default_values, 3, 'Function parameters contain type annotation and default values', 'ERROR'),
    ValidationModel(check_relevance, 4, 'Ideally, function should be relevant to existing functions or related to finance domain', 'WARNING'),
    ValidationModel(check_parameters, 5, 'Ideally, function should have more than 2 parameters', 'WARNING'),
    ValidationModel(check_input_output_stream, 6, 'Ideally, function should have both input and output streams', 'WARNING'),
]

validation_rules.sort(key=lambda x: x.priority)



def validate_func(module):
    validate_result = []

    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            # print(f'Function: {name}')
            # print(f'Args: {inspect.getfullargspec(obj)}')
            # print(f'Doc: {inspect.getdoc(obj)}')
            # print(f'Annotations: {obj.__annotations__}')
            # print()

            for validation_model in validation_rules:
                validate_result.append(validation_model.function(name, obj))
    
    return validate_result

def load_module(file_path):
    spec = importlib.util.spec_from_file_location('module', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def function_validation_result(file_path):
    module = load_module(file_path)

    validation_result = validate_func(module)
    flag = True
    message = ""
    for result in validation_result:
        if result.get_result() == False:
            message += str(result)
            if result.issue_type == 'ERROR':
                flag = False
    if flag:
        message = "All functions are valid"
    
    return flag, message
