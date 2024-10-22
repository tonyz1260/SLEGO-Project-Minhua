import os
import ast
import sys
import subprocess
import platform
import logging
import json
import re
import time
import inspect
import itertools
import importlib
from datetime import datetime
from typing import Dict, Any


# Install required packages if they are not already installed
def check_and_install_packages():
    # required_packages = ['panel', 'param', 'pandas', 'kglab', 'pyvis', 'rdflib']
    required_packages = ['panel', 'param', 'pandas', 'pyvis', 'rdflib']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

check_and_install_packages()

import panel as pn
import param
import pandas as pd
# import kglab
from pyvis.network import Network
from rdflib import URIRef

# Import recommender module
import recommender as rc
from validate_func import *
from function_generator import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Panel extensions
pn.extension('ace', 'jsoneditor', 'tabulator', sizing_mode='stretch_both')




class SLEGOApp:
    def __init__(self, config: Dict[str, Any]):
        logger.info("Initializing SLEGOApp...")
        self.config = config
        self.folder_path = config['folder_path']
        self.functionspace = config['functionspace']
        self.dataspace = config['dataspace']
        self.recordspace = config['recordspace']
        self.knowledgespace = config['knowledgespace']
        self.ontologyspace = config['ontologyspace']  # New: Add ontologyspace to config

        # Log configuration paths
        logger.info("Configuration paths:")
        logger.info("  folder_path: %s", self.folder_path)
        logger.info("  functionspace: %s", self.functionspace)
        logger.info("  dataspace: %s", self.dataspace)
        logger.info("  recordspace: %s", self.recordspace)
        logger.info("  knowledgespace: %s", self.knowledgespace)
        logger.info("  ontologyspace: %s", self.ontologyspace)

        self.func_file_path = 'func.py'
        self.initialize_widgets()
        self.setup_func_module()
        self.setup_event_handlers()
        self.create_layout()
        self.create_template()
        self.full_func_mapping_path = 'full_func.json'
        self.full_func_file_path = 'full_func.py'
        self.setup_full_func_mapping()
        self.output_text.value = '\nHello!\n\nWelcome to SLEGO - A Platform for Collaborative and Modular Data Analytics.\n\nPlease select the modules and functions to get started.\n\n'


    def initialize_widgets(self):
        logger.info("Initializing widgets...")
        # Ensure functionspace exists
        if not os.path.exists(self.functionspace):
            os.makedirs(self.functionspace)
            logger.info(f"Created functionspace directory: {self.functionspace}")

        # Include all .py files in the functionspace directory
        self.py_files = [f for f in os.listdir(self.functionspace) if f.endswith('.py')]

        if not self.py_files:
            logger.warning("No .py modules found in the functionspace.")
        else:
            logger.info(f"Python files available for selection: {self.py_files}")

            
        # Set default selected modules
        default_modules = ['func_vix.py', 'func_yfinance.py', 'Minhua_trial.py']
        default_selected = [module for module in default_modules if module in self.py_files]
        self.funcfilecombo = pn.widgets.MultiChoice(
            name='Select Module(s)',
            value=default_selected,
            options=self.py_files,
            height=100
        )

        self.compute_btn = pn.widgets.Button(name='Compute', height=50, button_type='primary')
        self.savepipe_btn = pn.widgets.Button(name='Save Pipeline', height=35)
        self.pipeline_text = pn.widgets.TextInput(value='', placeholder='Input Pipeline Name', height=35)
        self.json_toggle = pn.widgets.Toggle(name='Input mode: text or form', height=35, button_type='warning')
        self.json_editor = pn.widgets.JSONEditor(value={}, mode='form')
        self.input_text = pn.widgets.TextAreaInput(value='', placeholder='input the parameters')

        self.progress_text = pn.widgets.TextAreaInput(
            value='', 
            placeholder='Input your analytics query here', 
            name='User query inputs for recommendation or SPARQL:', 
            height=150
        )

        self.output_text = pn.widgets.TextAreaInput(
            value='', 
            placeholder='Results will be shown here', 
            name='System output message:'
        )
        
        # Added missing widgets with specified heights
        self.recommendation_btn = pn.widgets.Button(
            name='Get Recommendation', 
            height=35, 
            button_type='success'
        )
        self.recomAPI_text = pn.widgets.TextInput(
            value='', 
            placeholder='Your AI API key', 
            height=35
        )

        self.func_generator_btn = pn.widgets.Button(name='Generate Function', height=35, button_type='success')

        # File management widgets
        self.folder_select = pn.widgets.Select(
            name='Select Folder',
            options=[item for item in os.listdir(self.folder_path) 
                     if os.path.isdir(os.path.join(self.folder_path, item))] + ['/'],
            value='dataspace',
            height=50
        )
        self.file_text = pn.widgets.TextInput(
            value='/dataspace', 
            placeholder='Input the file name', 
            height=35
        )
        
        self.filefolder_confirm_btn = pn.widgets.Button(name='Confirm', height=35)
        self.file_view = pn.widgets.Button(name='View', height=35)
        self.file_download = pn.widgets.Button(name='Download', height=35)
        self.file_upload = pn.widgets.Button(name='Upload', height=35)
        self.file_input = pn.widgets.FileInput(name='Upload file', height=35, width=230)
        self.file_delete = pn.widgets.Button(name='Delete', height=35)
        self.file_table = self.create_file_table()

        self.widget_tab = pn.Tabs(
            ('JSON Input', self.json_editor), 
            ('Text Input', self.input_text)
        )
        self.ontology_btn = pn.widgets.Button(name='Show Ontology', height=35)

        self.rules_popup = pn.Card(pn.pane.Markdown("## Modal Title\nThis is the content inside the modal."), title="Modal", width=80, height=80, header_background="lightgray")
        self.rules_button = pn.widgets.Button(name="", icon="info-circle", width=10, height=35, button_type="light")
        # self.funccombo = pn.widgets.MultiChoice(name='Functions:', height=150)
        # Placeholder for funccombo
        self.funccombo_pane = pn.Column()
        logger.info("Widgets initialized.")


    # def setup_func_module(self):
    #     self.delete_func_file()
    #     self.create_func_file()
        
    #     # Optionally clear the module from cache
    #     if 'func' in sys.modules:
    #         del sys.modules['func']
        
    #     import func
    #     importlib.reload(func)
    #     self.func = func
        
    #     # Update self.funccombo with the new functions
    #     self.funccombo.options = self.get_functions_from_module(self.func)

    def setup_func_module(self):
        logger.info("Setting up func module...")
        selected_modules = self.funcfilecombo.value  # Get the selected modules
        if not selected_modules:
            logger.error("No modules selected. Please ensure desired modules exist in the functionspace.")
            return

        self.update_func_module(selected_modules)

    def update_func_module(self, module_names):
        logger.info(f"Updating functions for selected modules: {module_names}")
        if not module_names:
            self.funccombo_pane.objects = []
            self.output_text.value = "No modules selected."
            return

        self.modules = {}
        self.funcs = {}

        # Import selected modules dynamically
        for module_name in module_names:
            module_path = os.path.join(self.functionspace, module_name)
            if not os.path.exists(module_path):
                logger.warning(f"Module file {module_path} does not exist.")
                continue

            # Dynamically import the module
            spec = importlib.util.spec_from_file_location(module_name[:-3], module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.modules[module_name] = module

            # Get functions from the module
            module_functions = self.get_functions_from_module(module, module_name)
            self.funcs.update(module_functions)

        # Update function combo box
        self.funccombo = self.create_multi_select_combobox(self.funcs)
        self.funccombo_pane.objects = [self.funccombo]
        # Set up event handler for the new funccombo
        self.funccombo.param.watch(self.funccombo_change, 'value')
        logger.info("Function combobox updated based on the selected modules.")

    def setup_full_func_mapping(self):
        if os.path.exists(self.full_func_mapping_path):
            os.remove(self.full_func_mapping_path)

        if os.path.exists(self.full_func_file_path):
            os.remove(self.full_func_file_path)

        # Create a full function mapping file
        # key is the filename and value is a list of functions (name) in the file
        full_func_mapping = {}
        for py_file in self.py_files:
            file_path = os.path.join(self.functionspace, py_file)
            function_names = self.get_all_func_names(file_path)
            full_func_mapping[py_file] = function_names

            # also add all functions to the full_func.py file
            with open(self.full_func_file_path, 'a') as func_file:
                with open(file_path, 'r') as file:
                    func_file.write(file.read() + '\n')
        
        with open(self.full_func_mapping_path, 'w') as file:
            json.dump(full_func_mapping, file, indent=4)


    def get_all_func_names(self, filepath):
        with open(filepath, 'r') as file:
            content = file.read()
        
        # Parse the file content into an AST
        tree = ast.parse(content)
        
        # Extract all function definitions (ast.FunctionDef) from the tree
        function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
        return function_names

    # def delete_func_file(self):
    #     if os.path.exists(self.func_file_path):
    #         os.remove(self.func_file_path)

    # def create_func_file(self):
    #     with open(self.func_file_path, 'w') as func_file:
    #         for py_file in self.funcfilecombo.value:
    #             file_path = os.path.join(self.functionspace, py_file)
    #             with open(file_path, 'r') as file:
    #                 func_file.write(file.read() + '\n')

    def create_file_table(self):
        logger.info("Creating file table...")
        selected_folder_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'))
        logger.info(f"Selected folder path: {selected_folder_path}")
        if os.path.exists(selected_folder_path):
            file_list = os.listdir(selected_folder_path)
            df_file = pd.DataFrame(file_list, columns=['Filter Files :'])
            logger.info(f"Files in {selected_folder_path}: {file_list}")
            return pn.widgets.Tabulator(df_file, header_filters=True, show_index=False)
        else:
            logger.warning(f"Folder {selected_folder_path} does not exist.")
            return pn.widgets.Tabulator(pd.DataFrame(), header_filters=True, show_index=False)
        

    def setup_event_handlers(self):
        logger.info("Setting up event handlers...")
        self.funcfilecombo.param.watch(self.funcfilecombo_change, 'value')
        # Remove old funccombo watcher if it exists
        if hasattr(self, 'funccombo'):
            self.funccombo.param.watch(self.funccombo_change, 'value')
        self.input_text.param.watch(self.input_text_change, 'value')
        self.json_toggle.param.watch(self.json_toggle_clicked, 'value')
        self.json_editor.param.watch(self.json_editor_change, 'value')
        self.compute_btn.on_click(self.compute_btn_clicked)
        self.savepipe_btn.on_click(self.save_pipeline)
        self.filefolder_confirm_btn.on_click(self.on_filefolder_confirm_btn_click)
        self.file_view.on_click(self.on_file_buttons_click)
        self.file_download.on_click(self.on_file_buttons_click)
        self.file_upload.on_click(self.file_upload_click)
        self.file_delete.on_click(self.on_file_buttons_click)
        self.folder_select.param.watch(self.folder_select_changed, 'value')
        self.ontology_btn.on_click(self.ontology_btn_click)

        # Added event handler for recommendation button
        self.recommendation_btn.on_click(self.recommendation_btn_clicked)
        self.func_generator_btn.on_click(self.func_generator_btn_click)
        logger.info("Event handlers set up.")


    def create_layout(self):
        logger.info("Creating layout...")
        widget_input = pn.Column(
            pn.layout.Divider(height=10, margin=(10)), 
            self.widget_tab
        )
        widget_btns = pn.Row(self.savepipe_btn, self.pipeline_text, self.ontology_btn)
        widget_updownload = pn.Column(
            pn.Row(self.file_view, self.file_download),
            pn.Row(self.file_input, self.rules_button, width=280, height=50),
            pn.Row(self.file_upload, self.file_delete)
        )
        widget_files = pn.Column(
            self.folder_select,
            pn.Row(self.file_text, self.filefolder_confirm_btn),
            pn.layout.Divider(height=10, margin=(10)),
            self.file_table,
            widget_updownload,
            width=300, scroll=True
        )
        widget_funcsel = pn.Column(
            self.funcfilecombo, 
            self.funccombo_pane,  # Use the placeholder here
            self.compute_btn, 
            widget_btns,
            min_width=300
        )

        # Added recommendation widgets to the layout
        widget_recom = pn.Row(self.recommendation_btn, self.func_generator_btn)
        self.app = pn.Row(
            widget_files,
            pn.Column(widget_funcsel, widget_input, min_width=600, scroll=True),
            pn.Column(
                self.recomAPI_text,
                self.progress_text,
                widget_recom,
                pn.layout.Divider(height=10, margin=(10)), 
                self.output_text
            ), 
            min_width=600, scroll=True
        )
        logger.info("Layout created.")

    def funcfilecombo_change(self, event):
        logger.info(f"funcfilecombo changed: {event.new}")
        selected_modules = event.new
        self.update_func_module(selected_modules)

    def refresh_funcfilecombo(self):
        # Include all .py files in the functionspace directory
        self.py_files = [f for f in os.listdir(self.functionspace) if f.endswith('.py')]
        self.funcfilecombo.options = self.py_files
        logger.info(f"Refreshed funcfilecombo with {len(self.py_files)} files.")


    def get_functions_from_module(self, module, module_name):
        functions = {}
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module.__name__ and not name.startswith('_'):
                func_key = f"{module_name[:-3]}.{name}"
                functions[func_key] = obj
        return functions

    def funccombo_change(self, event):
        logger.info(f"funccombo changed: {event.new}")
        self.output_text.value = ''
        list_funcs = self.funccombo.value
        logger.info(f"Selected functions: {list_funcs}")
        list_params = []
        for func_key in list_funcs:
            try:
                func = self.funcs[func_key]
                params = self.extract_parameter(func)
                list_params.append(params)
            except Exception as e:
                logger.error(f"Error extracting parameters for function {func_key}: {e}")
                list_params.append({})
        funcs_params = dict(zip(list_funcs, list_params))
        formatted_data = json.dumps(funcs_params, indent=4)
        self.json_editor.value = funcs_params
        self.input_text.value = formatted_data
        self.output_text.value = self.get_doc_string(formatted_data)

    def input_text_change(self, event):
        logger.info("Input text changed.")
        text = re.sub(r'\bfalse\b', 'False', self.input_text.value, flags=re.IGNORECASE)
        text = text.replace("'", '"')
        try:
            pipeline_dict = json.loads(text)
            pipeline_dict_json = json.dumps(pipeline_dict, indent=4)
            self.input_text.value = pipeline_dict_json
            self.json_editor.value = json.loads(pipeline_dict_json)
            self.output_text.value += '\nInput changed!'
        except ValueError as e:
            self.output_text.value += f'\nError parsing input: {e}'
            logger.error(f"Error parsing input text: {e}")

    def json_toggle_clicked(self, event):
        logger.info(f"JSON toggle clicked: {event.new}")
        self.widget_tab.active = 1 if event.new else 0

    def json_editor_change(self, event):
        logger.info("JSON editor changed.")
        text = json.dumps(self.json_editor.value, indent=4)
        self.input_text.value = text

    def recommendation_btn_clicked(self, event):
        logger.info("Recommendation button clicked.")
        self.output_text.value = 'Asking AI for recommendation: \n'
        user_pipeline = self.json_editor.value
        user_query = self.progress_text.value
        db_path = os.path.join(self.folder_path, 'KB.db')
        openai_api_key = self.recomAPI_text.value

        try:
            response_text = rc.pipeline_recommendation(db_path, user_query, user_pipeline, openai_api_key)
            self.output_text.value += response_text
            self.output_text.value += '\n\n=================================\n'
            response_text = rc.pipeline_parameters_recommendation(user_query, response_text, openai_api_key)

            text = str(response_text)
            text = re.sub(r"\b(false|False)\b", '"false"', text, flags=re.IGNORECASE)

            self.output_text.value += response_text

            services = json.loads(response_text)
            keys = list(services.keys())
            self.funccombo.value = keys

            rec_string = json.dumps(services, indent=4)
            self.json_editor.value = services
            logger.info("Recommendation process completed.")
        except Exception as e:
            self.output_text.value += f"\nError during recommendation: {e}"
            logger.error(f"Error during recommendation: {e}")

    def func_generator_btn_click(self, event):
        logger.info("Function generator button clicked.")
        query = self.progress_text.value
        self.output_text.value = 'Generating function from query...'
        self.output_text.value = '\n\nBelow is the original query:\n\n' + query + '\n\n'
        generated_functions = generate_function(query)

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + 'auto_generated_function.py'

        temp_file_path = self.folder_path + '/temp/' + filename
        with open(temp_file_path, 'w') as f:
            f.write(generated_functions)

        flag, message, proposed_correction = function_validation_result(temp_file_path)

        folder = self.folder_path + '/functionspace'
        file_path = folder + '/' + filename

        self.output_text.value += '\n\n##################Generated & uploaded successfully!##############\n\n'
        self.output_text.value += f'The following function(s) have been generated and uploaded to SLEGO functionspace with filename: {filename}.\n'
        self.output_text.value += 'All functions are also validated and corrected if necessary.\n\n'
        self.output_text.value += message

        # add the generated functions' import statement to the file_path
        import_statements = extract_import_statements(temp_file_path)

        with open(file_path, 'w') as f:
            for statement in import_statements:
                f.write(statement)
                f.write('\n')
            f.write('\n')
            for _, correction in proposed_correction.items():
                f.write(correction)
                f.write('\n\n')

        if flag is False:
            
            self.output_text.value += "GENERATED FUNCTION WITH MODIFICATION\n\n"
            
        else:
            self.output_text.value += "NO MODIFICATION NEEDED\n\n"
            
        self.refresh_file_table()
        self.refresh_funcfilecombo()

        os.remove(temp_file_path)


    def compute_btn_clicked(self, event):
        logger.info("Compute button clicked.")
        self.progress_text.value = 'Computing...'
        pipeline_dict = self.json_editor.value
        self.output_text.value = ''
        logger.info(f"Pipeline dict: {pipeline_dict}")

        for func_key, parameters in pipeline_dict.items():
            logger.info(f"Computing {func_key} with parameters {parameters}")
            self.progress_text.value = f'Computing {func_key}...'
            try:
                start_time = time.time()
                function = self.funcs[func_key]
                result = function(**parameters)
                result_string = self.output_formatting(result)
                compute_time = time.time() - start_time

                self.output_text.value += f"\n===================={func_key}====================\n\n"
                self.output_text.value += f"Function computation Time: {compute_time:.4f} seconds\n\n"
                self.output_text.value += result_string
                logger.info(f"Function {func_key} computed successfully.")
            except Exception as e:
                self.output_text.value += f"\n===================={func_key}====================\n\n"
                self.output_text.value += f"Error occurred: {str(e)}\n"
                logger.error(f"Error computing {func_key}: {e}")

        self.save_record('recordspace', pipeline_dict)
        self.progress_text.value = 'Done!'
        self.on_filefolder_confirm_btn_click(None)
        self.refresh_file_table()

    def output_formatting(self, result):
        max_rows = 10
        max_words = 500
        max_chars = 2000

        final = ""

        # Check if the result is a pandas DataFrame
        if isinstance(result, pd.DataFrame):
            limited_df = result.head(max_rows)  # Limit to the first max_rows rows
            df_string = limited_df.to_string(index=False)
            final = df_string[:max_chars]  # Limit to max_chars characters
            if len(df_string) > max_chars:
                final += "\n... (truncated)"
        
        # Handle lists or dictionaries
        elif isinstance(result, (list, dict)):
            result_string = str(result)
            final = result_string[:max_chars]  # Limit to max_chars characters
            if len(result_string) > max_chars:
                final += "\n... (truncated)"

        else:
            result_string = str(result)
            words_iterator = iter(result_string.split())
            first_x_words = itertools.islice(words_iterator, max_words)
            truncated_result = " ".join(first_x_words)
            final = truncated_result[:max_chars]  # Limit to max_chars characters
            if len(truncated_result) > max_chars:
                final += "\n... (truncated)"

        return final

    def save_pipeline(self, event):
        logger.info("Save pipeline button clicked.")
        pipeline_name = self.pipeline_text.value if self.pipeline_text.value else '__'
        text = re.sub(r'\bfalse\b', 'False', self.input_text.value, flags=re.IGNORECASE)
        try:
            data = json.loads(text)
            self.save_record('knowledgespace', data, pipeline_name)
            self.on_filefolder_confirm_btn_click(None)
            self.output_text.value = f'Pipeline {pipeline_name} saved to knowledgespace!'
        except Exception as e:
            logger.error(f"Error saving pipeline: {e}")
            self.output_text.value += f"\nError saving pipeline: {e}"
        

    def file_upload_click(self, event):
        if self.file_input.filename:
            filename = self.file_input.filename
            self.output_text.value = f'Uploading {filename}...'

            file_content = self.file_input.value

            temp_file_path = self.folder_path + '/temp/' + filename
            with open(temp_file_path, 'wb') as f:
                f.write(file_content)

            flag, message, proposed_correction = function_validation_result(temp_file_path)
            
            folder = self.folder_path + '/' + self.file_text.value
            file_path = folder + '/' + filename

            # If a file already exists with the same name, add a timestamp to the filename
            if os.path.exists(file_path):
                filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + filename
                file_path = folder + '/' + filename
                self.output_text.value += f'File with the same name already exists. Renaming to {filename}...'

            # add the generated functions' import statement to the file_path
            import_statements = extract_import_statements(temp_file_path)
            
            with open(file_path, 'w') as f:
                for statement in import_statements:
                    f.write(statement)
                    f.write('\n')
                f.write('\n')
                for _, correction in proposed_correction.items():
                    f.write(correction)
                    f.write('\n\n')

            if flag is False:
                self.output_text.value = "\n\n##################File Upload Success BUT with modification##################\n\n" + message
                                
            else:

                self.output_text.value += f'\n\n##################{filename} uploaded successfully!##############\n\n' + message
                
            self.refresh_file_table()
            self.refresh_funcfilecombo()

            os.remove(temp_file_path)
        else:
            self.output_text.value = 'Please select a file to upload!'

    def on_file_buttons_click(self, event):
        logger.info(f"File button '{event.obj.name}' clicked.")
        self.output_text.value = ''
        file_list = self.file_table.selected_dataframe['Filter Files :'].tolist()
        if file_list:
            if event.obj.name == 'View':
                for filename in file_list:
                    file_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'), filename)
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as file:
                            content = file.read()
                        self.output_text.value += f"\n===== {filename} =====\n{content}\n"
                    else:
                        self.output_text.value += f"\nFile {filename} does not exist."
            elif event.obj.name == 'Download':
                self.output_text.value = 'Download functionality is not implemented.'
            elif event.obj.name == 'Upload' and not self.file_input.filename:
                self.output_text.value = 'Please use the file input widget to upload!'
            elif event.obj.name == 'Delete':
                self.output_text.value = 'Delete functionality is not implemented for safety reasons.'
        else:
            self.output_text.value = 'Please select a file to perform the action.'

    def on_filefolder_confirm_btn_click(self, event):
        logger.info("File folder confirm button clicked.")
        selected_folder_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'))
        if os.path.exists(selected_folder_path):
            file_list = os.listdir(selected_folder_path)
            df_file = pd.DataFrame(file_list, columns=['Filter Files :'])
            self.file_table.value = df_file
            logger.info(f"Updated file table with files from {selected_folder_path}")
        else:
            logger.warning(f"Folder {selected_folder_path} does not exist.")
            self.file_table.value = pd.DataFrame()

    def folder_select_changed(self, event):
        logger.info(f"Folder selected: {event.new}")
        self.file_text.value = '/' + str(self.folder_select.value)
        self.on_filefolder_confirm_btn_click(None)

    def get_doc_string(self, pipeline_text):
        output = ''
        data = json.loads(pipeline_text)
        for func_key in data.keys():
            output += f"===== {func_key} =====\n"
            try:
                func = self.funcs[func_key]
                doc = func.__doc__
                output += doc if doc else 'No docstring available.\n'
            except AttributeError:
                output += 'Function not found.\n'
        return output

    @staticmethod
    def is_colab_runtime():
        try:
            import google.colab
            return True
        except ImportError:
            return False

    def save_record(self, space_key, data, filename=None):
        space = self.config.get(space_key)
        if not space:
            logger.error(f"Invalid space key: {space_key}")
            return

        if filename is None:
            filename = datetime.now().strftime("record_%Y%m%d_%H%M%S.json")
        else:
            filename = filename + '.json'

        full_path = os.path.join(space, filename)
        try:
            with open(full_path, "w") as file:
                json.dump(data, file, indent=4)
            logger.info(f"Record saved to {full_path}")
            self.refresh_file_table()
        except Exception as e:
            logger.error(f"Error saving record: {e}")
            self.output_text.value += f"\nError saving record: {e}"

    def refresh_file_table(self):
        logger.info("Refreshing file table...")
        selected_folder_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'))
        if os.path.exists(selected_folder_path):
            file_list = os.listdir(selected_folder_path)
            df_file = pd.DataFrame(file_list, columns=['Filter Files :'])
            self.file_table.value = df_file
        else:
            self.file_table.value = pd.DataFrame()

    def create_multi_select_combobox(self, funcs):
        options = list(funcs.keys())
        multi_combobox = pn.widgets.MultiChoice(name='Functions:', options=options, height=150)
        return multi_combobox

    # def get_functions_from_module(self, target_module):
    #     """
    #     Creates a multi-select combobox with all functions from the target_module.
    #     """
    #     module_name = target_module.__name__
    #     functions = [name for name, obj in inspect.getmembers(target_module, inspect.isfunction)
    #                  if obj.__module__ == module_name and not name.startswith('_')]
    #     self.output_text.value += f"\nLoaded {len(functions)} functions from {module_name} module.\n"
    #     return functions

    def extract_parameter(self, func):
        """
        Extracts the names and default values of the parameters of a function as a dictionary.
        """
        signature = inspect.signature(func)
        parameters = signature.parameters
        parameter_dict = {}
        for name, param in parameters.items():
            if param.default != inspect.Parameter.empty:
                parameter_dict[name] = param.default
            else:
                parameter_dict[name] = None
        return parameter_dict
    
    def ontology_btn_click(self, event):
        logger.info("Ontology button clicked.")
        input_file_path = os.path.join(self.ontologyspace, "hfd.ttl")
        output_file_path = os.path.join(self.ontologyspace, "hfd_visualization.html")
        self.visualize_rdf_graph(input_file_path, output_file_path)

        if os.path.exists(output_file_path):
            if self.is_colab_runtime():
                from IPython.display import IFrame, display
                display(IFrame(src=output_file_path, width='100%', height='600px'))
            else:
                import webbrowser
                webbrowser.open_new_tab(f'file://{os.path.abspath(output_file_path)}')
        else:
            logger.error(f"The file {output_file_path} does not exist.")

    # def visualize_rdf_graph(self, input_file_path, output_file_path):
    #     logger.info("Visualizing RDF graph...")
    #     try:
    #         kg = kglab.KnowledgeGraph()
    #         kg.load_rdf(input_file_path)
    #         net = Network(
    #             notebook=True, height="1000px", width="100%",
    #             bgcolor="#ffffff", font_color="black",
    #             directed=True, cdn_resources='remote'
    #         )

    #         # Define visualization options and styles as per your previous code
    #         net.set_options("""
    #         var options = {
    #             "nodes": {
    #                 "shape": "box",
    #                 "size": 30,
    #                 "font": {
    #                     "size": 14,
    #                     "face": "Tahoma"
    #                 }
    #             },
    #             "edges": {
    #                 "arrows": {
    #                     "to": {
    #                         "enabled": true,
    #                         "scaleFactor": 1
    #                     }
    #                 },
    #                 "smooth": {
    #                     "type": "continuous"
    #                 }
    #             },
    #             "layout": {
    #                 "hierarchical": {
    #                     "enabled": true,
    #                     "levelSeparation": 250,
    #                     "nodeSpacing": 200,
    #                     "treeSpacing": 300,
    #                     "blockShifting": true,
    #                     "edgeMinimization": true,
    #                     "parentCentralization": true,
    #                     "direction": "LR",
    #                     "sortMethod": "hubsize"
    #                 }
    #             },
    #             "physics": {
    #                 "enabled": false
    #             }
    #         }
    #         """)

    #         for subject, predicate, obj in kg.rdf_graph().triples((None, None, None)):
    #             if isinstance(subject, URIRef):
    #                 self.__add_node(net, kg, subject)
    #             if isinstance(obj, URIRef):
    #                 self.__add_node(net, kg, obj)
    #             if isinstance(predicate, URIRef):
    #                 edge_label = predicate.split("/")[-1]
    #                 net.add_edge(str(subject), str(obj), label=edge_label)

    #         net.save_graph(output_file_path)
    #         logger.info(f"RDF graph saved to {output_file_path}")
    #     except Exception as e:
    #         logger.error(f"Error visualizing RDF graph: {e}")
    #         self.output_text.value += f"\nError visualizing RDF graph: {e}"

    def __add_node(self, net, kg, node):
        label = node.split("/")[-1]
        node_class = kg.rdf_graph().value(node, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"))
        if node_class:
            class_name = node_class.split("/")[-1]
            style = {"color": "gray", "shape": "ellipse", "size": 10}
        else:
            style = {"color": "gray", "shape": "ellipse", "size": 10}
        net.add_node(
            str(node), 
            label=label, 
            color=style["color"], 
            shape=style["shape"], 
            size=style["size"]
        )
    
    def create_validation_rules_message(self):
        message = "Microservice Rules:\n\n"
        error_rule_message = ""
        warning_rule_message = ""
        for rule in validation_rules:
            if rule.issue_type == 'ERROR':
                error_rule_message += f"{rule.description}\n"
            if rule.issue_type == 'WARNING':
                warning_rule_message += f"{rule.description}\n"
        message = f"**The following rules will lead to unsuccessful upload if not followed**\n\n{error_rule_message}\n\n"
        message += f"**The following rules are recommended to be followed**\n\n{warning_rule_message}"
        return message
    
    def create_template(self):
        template = pn.template.MaterialTemplate(
            title='SLEGO - Software Lego: A Collaborative and Modular Architecture for Data Analytics',
            sidebar=[],
        )

        template.modal.append(self.create_validation_rules_message())

        template.main.append(self.app)
        self.template = template

    def toggle_rule(self, event):
        self.template.open_modal()

    def run(self):
        logger.info("Running the app...")
        if not self.is_colab_runtime():
            self.rules_button.on_click(self.toggle_rule)

            self.template.main.append(self.app)
            self.template.show()
            logger.info("App is running in non-Colab environment.")
        else:
            from IPython.display import display
            display(self.app)
            logger.info("App is running in Colab environment.")

