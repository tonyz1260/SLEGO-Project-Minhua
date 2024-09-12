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
        
def api_call(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY2)

    # Create a chat completion request
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a python programmer and tester."},
            {"role": "user", "content": f"{prompt}"},
        ],
    )

    reply = response.choices[0].message.content

    print(response.choices[0].message.content)

    return reply

def correction_proposal_gpt(name, obj, issue_type, issue_message):
    prompt = "You are now a software engineer and your task is to propose a potential correction for the function that is provided to you. \
    You are given a function that is written in Python and the function contains certain issues that need to be corrected. \
    Based on the issue type and message that is provided, you need to propose a potential correction for the function. \
    You will be provided with the function source code and the issue type and message that you need to address. \
    Your final return response should include the proposed correction for the function and must be concise (less than 15 words ideally). \
    You must not add any additional text or characters to the response. From here, I will provide you with the function source code and the issue type and message that you need to address."

    prompt += f"\n\nFunction: {name}\nSource Code: {inspect.getsource(obj)}\n"
    prompt += f"\n\nIssue Type: {issue_type}\nIssue Message: {issue_message}\n"

    response = api_call(prompt)

    return response

# Warnings -------------------------------------------------------------------------------
# Check for function relevance
warning_response_cache = {}

def warning_api_call(name, obj):
    if name in warning_response_cache:
        return warning_response_cache[name]

    prompt = "You are now a software engineer and your task is to assess the relevance of the functions that are provided to you. \
    You are given a function that is written in Python and you need to determine whether the function is relevant to the existing functions (docstrings of existing functions will be provided) and/or related to the finance domain. \
    In addition, you should also determine whether the function is meant to have the input and output parameters. \
    What that means is based on the actual function context, it may be a function that does webscraping thus does have the output path (to store webscraping result) but not the input path. \
    Or it may be a function that does not have the output path but does have the input path. \
    So your final return response should include the result and must strictly follow this format 'Relevance: [true/false]; Parameter: [input/output/both]' \
    You must not add any additional text or characters to the response. From here, I will provide you with the docstrings of existing functions that you need to assess."

    # iterate through the existing functions in func.py and provide the docstrings for the prompt
    for func_name, func_obj in inspect.getmembers(func):
        if inspect.isfunction(func_obj) and func_obj.__module__ == func.__name__:
            prompt += f"\n\nFunction: {func_name}\nDocstring: {inspect.getdoc(func_obj)}\n"

    # append the function (source_code) that needs to be assessed
    prompt += "Here is the function that you need to assess"
    prompt += f"\n\nFunction: {name}\nSource Code: {inspect.getsource(obj)}\n"

    response = api_call(prompt)
    warning_response_cache[name] = response

    return response


def check_relevance(name, obj):
    response = warning_api_call(name, obj)

    if 'Relevance: true' in response:
        return ValidationResult(name, True, 'Function is relevant to existing functions and related to finance domain', 'WARNING')
    else:
        return ValidationResult(name, False, 'Function is not relevant to existing functions or related to finance domain', 'WARNING')

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

    response = warning_api_call(name, obj)

    if 'Parameter: both' in response:
        if not has_input_stream or not has_output_stream:
            return ValidationResult(name, False, 'Function is expected to contain both input and output stream parameters', 'WARNING')
    elif 'Parameter: input' in response:
        if not has_input_stream:
            return ValidationResult(name, False, 'Function is expected to contain input stream parameter', 'WARNING')
    elif 'Parameter: output' in response:
        if not has_output_stream:
            return ValidationResult(name, False, 'Function is expected to contain output stream parameter', 'WARNING')
    
    return ValidationResult(name, True, 'Function has input and output stream', 'WARNING')

# Errors -------------------------------------------------------------------------------
def check_docstring(name, obj):
    if inspect.getdoc(obj) is None:
        return ValidationResult(name, False, 'Function does not have a docstring', 'ERROR')
    
    # TODO: Check whether the inspect.getdoc(obj) matches the function's actual behavior
    prompt = "You are now a software engineer and your task is to assess the docstring of the function that is provided to you. \
    You are given a function that is written in Python and you need to determine whether the function contains a docstring that accurately describes the function's behavior. \
    Based on the function's source code, you need to assess whether the docstring is accurate. \
    Your final return response should only be True or False and nothing else. \
    From here, I will provide you with the function source code and the docstring that you need to assess."

    prompt += f"\n\nFunction: {name}\nSource Code: {inspect.getsource(obj)}\n"
    prompt += f"\n\nDocstring: {inspect.getdoc(obj)}\n"

    response = api_call(prompt)

    if response == 'False':
        return ValidationResult(name, False, 'Function docString does not accurately describe the function behavior', 'WARNING')
    else:
        return ValidationResult(name, True, 'Function docString is valid', 'ERROR')

def check_annotations_and_default_values(name, obj):
    sig = inspect.signature(obj)
    parameters = sig.parameters

    validation_result_list = []

    for param in parameters.values():
        if param.annotation == inspect.Parameter.empty:
            validation_result_list.append(ValidationResult(name, False, f'Parameter {param.name} does not have annotation', 'ERROR'))
        if param.default == inspect.Parameter.empty:
            validation_result_list.append(ValidationResult(name, False, f'Parameter {param.name} does not have default value', 'ERROR'))

    if len(validation_result_list) > 0:
        return validation_result_list
    else:
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
    ValidationModel(check_input_output_stream, 5, 'Ideally, function should have both input and output streams', 'WARNING'),
]

validation_rules.sort(key=lambda x: x.priority)

def insert_validation_rule_in_place(rules, new_rule, new_priority):
    # Shift the priority of existing rules in place
    for rule in rules:
        if rule.priority >= new_priority:
            rule.priority += 1

    # Set the priority for the new rule and add it to the existing list
    new_rule.priority = new_priority
    rules.append(new_rule)

    # Sort in-place to maintain the order by priority
    rules.sort(key=lambda x: x.priority)

    return rules




def validate_func(module):
    validate_result = []

    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            print(f'Function: {name}')
            # print(f'Args: {inspect.getfullargspec(obj)}')
            # print(f'Doc: {inspect.getdoc(obj)}')
            # print(f'Annotations: {obj.__annotations__}')
            print()

            for validation_model in validation_rules:
                result = validation_model.function(name, obj)

                if isinstance(result, list):
                    for res in result:
                        if res.get_result() is False and res.issue_type == 'ERROR':
                            correction = correction_proposal_gpt(name, obj, res.issue_type, res.message)
                            print(correction)
                            res = ValidationResult(name, False, f'{res.message}\nProposed Correction: {correction}', 'ERROR')
                        validate_result.append(res)
                else:
                    # for error type, if the result is False, then propose a correction
                    if result.get_result() is False and result.issue_type == 'ERROR':
                        correction = correction_proposal_gpt(name, obj, result.issue_type, result.message)
                        print(correction)
                        result = ValidationResult(name, False, f'{result.message}\nProposed Correction: {correction}', 'ERROR')
                    validate_result.append(result)
    
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
    message, error_message, warning_message = "", "", ""
    for result in validation_result:
        if result.get_result() is False:
            if result.issue_type == 'ERROR':
                flag = False
                error_message += str(result)
            elif result.issue_type == 'WARNING':
                warning_message += str(result)
    message = error_message + "\n" + warning_message

    if flag:
        message = "All functions are valid"
    # print(flag, message)
    return flag, message


# print(function_validation_result('func_api.py'))