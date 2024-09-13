import os
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
    required_packages = ['panel', 'param', 'pandas', 'kglab', 'pyvis', 'rdflib']
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
import kglab
from pyvis.network import Network
from rdflib import URIRef

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
        self.ontologyspace = config['ontologyspace']
        self.func_file_path = os.path.join(os.getcwd(), 'func.py')

        # Log configuration paths
        logger.info(f"Configuration paths:")
        logger.info(f"  folder_path: {self.folder_path}")
        logger.info(f"  functionspace: {self.functionspace}")
        logger.info(f"  dataspace: {self.dataspace}")
        logger.info(f"  recordspace: {self.recordspace}")
        logger.info(f"  knowledgespace: {self.knowledgespace}")
        logger.info(f"  ontologyspace: {self.ontologyspace}")

        self.initialize_widgets()
        self.setup_func_module()
        self.setup_event_handlers()
        self.create_layout()
        logger.info("SLEGOApp initialized.")

    def initialize_widgets(self):
        logger.info("Initializing widgets...")
        # Ensure functionspace exists
        if not os.path.exists(self.functionspace):
            os.makedirs(self.functionspace)
            logger.info(f"Created functionspace directory: {self.functionspace}")

        self.py_files = [file for file in os.listdir(self.functionspace) if file.endswith('.py')]
        logger.info(f"Python files found in functionspace: {self.py_files}")

        self.funcfilecombo = pn.widgets.MultiChoice(
            name='Select Modules',
            value=self.py_files,  # Pre-select all available .py files
            options=self.py_files,
            height=150
        )
        self.compute_btn = pn.widgets.Button(name='Compute', height=50, button_type='primary')
        self.savepipe_btn = pn.widgets.Button(name='Save Pipeline', height=35)
        self.pipeline_text = pn.widgets.TextInput(value='', placeholder='Input Pipeline Name', height=35)
        self.json_toggle = pn.widgets.Toggle(name='Input mode: text or form', height=35, button_type='warning')
        self.json_editor = pn.widgets.JSONEditor(value={}, mode='form')
        self.input_text = pn.widgets.TextAreaInput(value='', placeholder='Input the parameters')
        self.progress_text = pn.widgets.TextAreaInput(value='', placeholder='Input your analytics query here', name='User query inputs for recommendation or SPARQL:', height=150)
        self.output_text = pn.widgets.TextAreaInput(value='', placeholder='Results will be shown here', name='System output message:')

        # File management widgets
        self.folder_select = pn.widgets.Select(
            name='Select Folder',
            options=[item for item in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, item))],
            value='dataspace',
            height=50
        )
        self.file_text = pn.widgets.TextInput(value='/dataspace', placeholder='Input the file name')
        self.filefolder_confirm_btn = pn.widgets.Button(name='Confirm')
        self.file_view = pn.widgets.Button(name='View')
        self.file_download = pn.widgets.Button(name='Download')
        self.file_upload = pn.widgets.Button(name='Upload')
        self.file_input = pn.widgets.FileInput(name='Upload file')
        self.file_delete = pn.widgets.Button(name='Delete')
        self.file_table = self.create_file_table()

        self.widget_tab = pn.Tabs(('JSON Input', self.json_editor), ('Text Input', self.input_text))
        self.ontology_btn = pn.widgets.Button(name='Show Ontology', height=35)
        logger.info("Widgets initialized.")

    def setup_func_module(self):
        logger.info("Setting up func module...")
        self.delete_func_file()
        self.create_func_file()

        # Ensure the directory containing func.py is in sys.path
        func_dir = os.path.abspath(os.path.dirname(self.func_file_path))
        if func_dir not in sys.path:
            sys.path.insert(0, func_dir)
            logger.info(f"Added {func_dir} to sys.path.")

        try:
            import func
            importlib.reload(func)
            self.func = func
            logger.info("func module imported and reloaded successfully.")
        except Exception as e:
            logger.error(f"Error importing func module: {e}")
            raise

        self.funccombo = self.create_multi_select_combobox(self.func)
        logger.info("func module set up.")

    def delete_func_file(self):
        if os.path.exists(self.func_file_path):
            try:
                os.remove(self.func_file_path)
                logger.info(f"Deleted existing {self.func_file_path}.")
            except Exception as e:
                logger.error(f"Error deleting {self.func_file_path}: {e}")

    def create_func_file(self):
        logger.info(f"Creating {self.func_file_path}...")
        try:
            with open(self.func_file_path, 'w') as func_file:
                for py_file in self.funcfilecombo.value:
                    file_path = os.path.join(self.functionspace, py_file)
                    logger.info(f"Adding contents of {file_path} to {self.func_file_path}.")
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as file:
                            func_file.write(file.read() + '\n')
                    else:
                        logger.warning(f"File {file_path} does not exist.")
            logger.info(f"{self.func_file_path} created.")
        except Exception as e:
            logger.error(f"Error creating {self.func_file_path}: {e}")
            raise

    def create_file_table(self):
        logger.info("Creating file table...")
        selected_folder_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'))
        logger.info(f"Selected folder path: {selected_folder_path}")
        if os.path.exists(selected_folder_path):
            file_list = os.listdir(selected_folder_path)
            df_file = pd.DataFrame(file_list, columns=['Files'])
            logger.info(f"Files in {selected_folder_path}: {file_list}")
            return pn.widgets.Tabulator(df_file, header_filters=True, show_index=False)
        else:
            logger.warning(f"Folder {selected_folder_path} does not exist.")
            return pn.widgets.Tabulator(pd.DataFrame(), header_filters=True, show_index=False)

    def setup_event_handlers(self):
        logger.info("Setting up event handlers...")
        self.funcfilecombo.param.watch(self.funcfilecombo_change, 'value')
        self.funccombo.param.watch(self.funccombo_change, 'value')
        self.input_text.param.watch(self.input_text_change, 'value')
        self.json_toggle.param.watch(self.json_toggle_clicked, 'value')
        self.json_editor.param.watch(self.json_editor_change, 'value')
        self.compute_btn.on_click(self.compute_btn_clicked)
        self.savepipe_btn.on_click(self.save_pipeline)
        self.filefolder_confirm_btn.on_click(self.on_filefolder_confirm_btn_click)
        self.file_view.on_click(self.on_file_buttons_click)
        self.file_download.on_click(self.on_file_buttons_click)
        self.file_upload.on_click(self.on_file_buttons_click)
        self.file_delete.on_click(self.on_file_buttons_click)
        self.folder_select.param.watch(self.folder_select_changed, 'value')
        self.ontology_btn.on_click(self.ontology_btn_click)
        logger.info("Event handlers set up.")

    def create_layout(self):
        logger.info("Creating layout...")
        widget_input = pn.Column(pn.layout.Divider(), self.widget_tab)
        widget_btns = pn.Row(self.savepipe_btn, self.pipeline_text, self.ontology_btn)
        widget_updownload = pn.Column(
            pn.Row(self.file_view, self.file_download),
            self.file_input,
            pn.Row(self.file_upload, self.file_delete)
        )
        widget_files = pn.Column(
            self.folder_select,
            pn.Row(self.file_text, self.filefolder_confirm_btn),
            self.file_table,
            widget_updownload,
            width=250,
            margin=(0, 20, 0, 0)
        )
        widget_funcsel = pn.Column(self.funcfilecombo, self.funccombo, self.compute_btn, widget_btns)
        self.app = pn.Row(
            widget_files,
            pn.Column(widget_funcsel, widget_input),
            pn.Column(self.progress_text, pn.layout.Divider(), self.output_text)
        )
        logger.info("Layout created.")

    def funcfilecombo_change(self, event):
        logger.info(f"funcfilecombo changed: {event.new}")
        self.setup_func_module()

    def funccombo_change(self, event):
        logger.info(f"funccombo changed: {event.new}")
        self.output_text.value = ''
        list_funcs = self.funccombo.value
        logger.info(f"Selected functions: {list_funcs}")
        list_params = []
        for func_name in list_funcs:
            try:
                params = self.extract_parameter(getattr(self.func, func_name))
                list_params.append(params)
            except Exception as e:
                logger.error(f"Error extracting parameters for function {func_name}: {e}")
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

    def compute_btn_clicked(self, event):
        logger.info("Compute button clicked.")
        self.progress_text.value = 'Computing...'
        pipeline_dict = self.json_editor.value
        self.output_text.value = ''
        logger.info(f"Pipeline dict: {pipeline_dict}")

        for function_name, parameters in pipeline_dict.items():
            logger.info(f"Computing {function_name} with parameters {parameters}")
            self.progress_text.value = f'Computing {function_name}...'
            try:
                start_time = time.time()
                function = getattr(self.func, function_name)
                result = function(**parameters)
                result_string = str(result)
                compute_time = time.time() - start_time

                self.output_text.value += f"\n===== {function_name} =====\n\n"
                self.output_text.value += f"Function computation time: {compute_time:.4f} seconds\n\n"
                self.output_text.value += result_string[:1000] + '... [truncated]' if len(result_string) > 1000 else result_string
                logger.info(f"Function {function_name} computed successfully.")
            except Exception as e:
                self.output_text.value += f"\n===== {function_name} =====\n\n"
                self.output_text.value += f"Error occurred: {str(e)}\n"
                logger.error(f"Error computing {function_name}: {e}")

        self.save_record('recordspace', pipeline_dict)
        self.progress_text.value = 'Done!'
        self.on_filefolder_confirm_btn_click(None)
        self.refresh_file_table()

    def save_pipeline(self, event):
        logger.info("Save pipeline button clicked.")
        pipeline_name = self.pipeline_text.value if self.pipeline_text.value else '__'
        text = re.sub(r'\bfalse\b', 'False', self.input_text.value, flags=re.IGNORECASE)
        try:
            data = json.loads(text)
            self.save_record('knowledgespace', data, pipeline_name)
            self.on_filefolder_confirm_btn_click(None)
        except Exception as e:
            logger.error(f"Error saving pipeline: {e}")
            self.output_text.value += f"\nError saving pipeline: {e}"

    def on_file_buttons_click(self, event):
        logger.info(f"File button '{event.obj.name}' clicked.")
        self.output_text.value = ''
        file_list = self.file_table.selected_dataframe['Files'].tolist()
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
            elif event.obj.name == 'Upload':
                self.output_text.value = 'Please use the file input widget to upload!'
            elif event.obj.name == 'Delete':
                self.output_text.value = 'Delete functionality is not implemented.'
        else:
            self.output_text.value = 'Please select a file to perform the action.'

    def on_filefolder_confirm_btn_click(self, event):
        logger.info("File folder confirm button clicked.")
        selected_folder_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'))
        if os.path.exists(selected_folder_path):
            file_list = os.listdir(selected_folder_path)
            df_file = pd.DataFrame(file_list, columns=['Files'])
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
        for func_name in data.keys():
            output += f"===== {func_name} =====\n"
            try:
                doc = getattr(self.func, func_name).__doc__
                output += doc if doc else 'No docstring available.\n'
            except AttributeError:
                output += 'Function not found.\n'
        return output

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
            df_file = pd.DataFrame(file_list, columns=['Files'])
            self.file_table.value = df_file
        else:
            self.file_table.value = pd.DataFrame()

    def create_multi_select_combobox(self, target_module):
        module_name = target_module.__name__
        functions = [name for name, obj in inspect.getmembers(target_module, inspect.isfunction)
                     if obj.__module__ == module_name and not name.startswith('_')]
        multi_combobox = pn.widgets.MultiChoice(name='Functions:', options=functions, height=150)
        return multi_combobox

    def extract_parameter(self, func):
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
        rdf_html_path = self.visualize_rdf_graph(input_file_path, output_file_path)

        if os.path.exists(rdf_html_path):
            if self.is_colab_runtime():
                from IPython.display import IFrame, display
                display(IFrame(src=rdf_html_path, width='100%', height='600px'))
            else:
                import webbrowser
                webbrowser.open_new_tab(f'file://{os.path.abspath(rdf_html_path)}')
        else:
            logger.error(f"The file {rdf_html_path} does not exist.")

    def visualize_rdf_graph(self, input_file_path, output_file_path):
        logger.info("Visualizing RDF graph...")
        try:
            kg = kglab.KnowledgeGraph()
            kg.load_rdf(input_file_path)
            net = Network(notebook=True, height="1000px", width="100%", bgcolor="#ffffff", font_color="black")
            net.from_nx(kg.get_nx_graph())
            net.save_graph(output_file_path)
            logger.info(f"RDF graph saved to {output_file_path}")
            return output_file_path
        except Exception as e:
            logger.error(f"Error visualizing RDF graph: {e}")
            self.output_text.value += f"\nError visualizing RDF graph: {e}"

    def run(self):
        logger.info("Running the app...")
        if not self.is_colab_runtime():
            template = pn.template.MaterialTemplate(
                title='SLEGO - Software Lego: A Collaborative and Modular Architecture for Data Analytics',
                sidebar=[],
            )
            template.main.append(self.app)
            template.show()
            logger.info("App is running in non-Colab environment.")
        else:
            from IPython.display import display
            display(self.app)
            logger.info("App is running in Colab environment.")

    @staticmethod
    def is_colab_runtime():
        return 'google.colab' in sys.modules

# At the end of your script, ensure that you have the following code to run the app:

# if __name__ == '__main__':
#     # Define your configuration dictionary
#     config = {
#         'folder_path': '/path/to/your/slegospace',  # Replace with your actual path
#         'functionspace': '/path/to/your/functionspace',
#         'dataspace': '/path/to/your/dataspace',
#         'recordspace': '/path/to/your/recordspace',
#         'knowledgespace': '/path/to/your/knowledgespace',
#         'ontologyspace': '/path/to/your/ontologyspace',
#     }

#     # Create and run the app
#     slego_app = SLEGOApp(config)
#     slego_app.run()
