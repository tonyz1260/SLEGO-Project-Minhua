import requests
import json
import os
import csv

def api_call(call_type: str, url: str, json_body: dict, json_structure: str, output_path: str):
    """
    Make an API call to the given URL with the given request type and data.

    :param request_type: The type of request to make (GET, POST, PUT, DELETE)
    :param url: The URL to make the request to
    :param data: The data to send with the request
    :param headers: The headers to send with the request
    :return: The response from the API
    """
    # Check the call type and make the corresponding API call
    if call_type.upper() == "GET":
        response = requests.get(url)
    elif call_type.upper() == "POST":
        response = requests.post(url, json=json_body)
    elif call_type.upper() == "PUT":
        response = requests.put(url, json=json_body)
    else:
        raise ValueError(f"Unsupported call type: {call_type}")
    
    # Raise an exception if the call was unsuccessful
    response.raise_for_status()
    
    # Parse the JSON response
    data = response.json()
    
    # Traverse the JSON structure to extract the relevant data
    if json_structure:
        keys = json_structure.split('/')
        for key in keys:
            data = data[key]
    
    # Save the data to a file if an output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)
    
    return data

def convert_json_to_csv(json_input_path: str, csv_path: str):
    """
    Convert a JSON file to a CSV file.

    :param json_input_path: The path to the JSON file to convert
    :param csv_path: The path to save the CSV file
    """
    # Load the JSON data
    with open(json_input_path, 'r') as infile:
        data = json.load(infile)
    
    # Extract the keys from the JSON data
    keys = data[0].keys()
    
    # Write the data to a CSV file
    with open(csv_path, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow(row)