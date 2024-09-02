from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, create_model
import inspect
from typing import get_origin, get_args, Union

app = FastAPI()

def generate_pydantic_model(func):
    sig = inspect.signature(func)
    fields = {}

    for param in sig.parameters.values():
        annotation = param.annotation

        if annotation is not inspect.Parameter.empty:
            fields[param.name] = (annotation, param.default)
        else:
            print(f"Parameter {param.name} in function {func.__name__} does not have a type annotation.")
            print(f"Parameter {param.name} in function {func.__name__} has annotation as {param.annotation}")
    # print(f"Fields: {fields}")
    return create_model(func.__name__ + "Model", **fields)

def create_route(func):
    model = generate_pydantic_model(func)

    async def route_func(params: model = Depends()):
        try:
            result = func(**params.dict())
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return route_func

def add_function_as_route(func, tag):
    route_func = create_route(func)
    print(f"Adding route for function: {func.__name__}")
    print(f"route_func: {route_func}")
    app.post(f"/{func.__name__}", tags=[tag])(route_func)

import os
import importlib.util
import tempfile

os.chdir('slegospace')

directory = 'functionspace'


def process_directory(directory):
    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.py'):  # Ensure it's a Python file
            # Create the full path to the file
            filepath = os.path.join(directory, filename)

            print("Processing file:", filename)

            # Read and preprocess the file to ignore imports
            with open(filepath, 'r') as file:
                lines = file.readlines()

            # Remove or comment out import statements
            modified_lines = lines
            # for line in lines:
            #     if not line.strip().startswith('import ') and not line.strip().startswith('from '):
            #         modified_lines.append(line)

            # Write the modified content to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode='w') as temp_file:
                temp_file.writelines(modified_lines)
                temp_filepath = temp_file.name

            try:
                # Dynamically load the module from the modified file
                spec = importlib.util.spec_from_file_location(filename[:-3], temp_filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Iterate over each function and do something with it
                for func_name, func in inspect.getmembers(module):
                    if inspect.isfunction(func) and func.__module__ == module.__name__:
                        if func_name != "preprocess_filling_missing_values":
                            print(f"Function name: {func_name}")
                            # print(f"Function object: {func}")
                            add_function_as_route(func, tag=filename[:-3])

            except Exception as e:
                # Handle possible exceptions during module loading
                print(f"Skipping {filename} due to an error: {str(e)}")
                continue

            finally:
                # Clean up the temporary file
                os.remove(temp_filepath)

@app.get("/")
def read_root():
    return {"message": "Welcome to the dynamically generated API! Available endpoints are based on the functions in your Python files."}



process_directory(directory)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)

