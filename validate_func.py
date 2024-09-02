import inspect
import importlib.util

class ValidationResult:
    def __init__(self, function_name: str, validation_result: bool, message: str):
        self.function_name = function_name
        self.validation_result = validation_result
        self.message = message

    def __str__(self):
        return f'Function: {self.function_name}\nValidation Result: {self.validation_result}\nMessage: {self.message}\n'
    
    def get_result(self) -> bool:
        return self.validation_result
    

# TODO: Have a dictionary for all of the rules, indicator
# 

def validate_func(module):
    validate_result = []

    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            # print(f'Function: {name}')
            # print(f'Args: {inspect.getfullargspec(obj)}')
            # print(f'Doc: {inspect.getdoc(obj)}')
            # print(f'Annotations: {obj.__annotations__}')
            # print()

            sig = inspect.signature(obj)
            parameters = sig.parameters

            if len(parameters) < 2:
                validate_result.append(ValidationResult(name, False, 'Function does not have enough parameters'))
                continue

            if inspect.getdoc(obj) is None:
                validate_result.append(ValidationResult(name, False, 'Function does not have a docstring'))
                continue

            has_input_stream = False
            has_output_stream = False
            for param in parameters.values():
                if param.annotation == str and param.default != inspect.Parameter.default:
                    if 'input' in param.name:
                        has_input_stream = True
                    if 'output' in param.name:
                        has_output_stream = True

            if not has_input_stream:
                validate_result.append(ValidationResult(name, False, 'Function does not have input stream or not configured correctly'))
                continue
            if not has_output_stream:
                validate_result.append(ValidationResult(name, False, 'Function does not have output stream or not configured correctly'))
                continue

            # TODO: On the platform, test the validation and show the message accordingly
            # TODO: Powerpoint, diagrams to show the workflow, how to validate the functions being valid

            # TODO: Check whether the inspect.getdoc(obj) matches the function's actual behavior
            # including the parameters, return type, and the description
            # Maybe accomplish by GPT call?

            # TODO: How to determine if the function is not meant to have input/output streams?
            # Define the input/output
            # Call OPENAI potentially? (last step)
            # Smaller Local LLM? (first step)

            # JSON Data Structure, time-series data with specific attributes, etc.

            # we give feedback on how to fix the function

            # Auto Edit

            validate_result.append(ValidationResult(name, True, 'Function is valid'))
    
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
            flag = False
    if flag:
        message = "All functions are valid"
    
    return flag, message
