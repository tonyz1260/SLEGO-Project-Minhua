import os
import ast
import panel as pn
import inspect
import json
import importlib
import re
import io
import time
import param
from datetime import datetime
import itertools
import pandas as pd
import recommender as rc

pn.extension()
pn.extension(sizing_mode='stretch_both')
pn.extension('ace', 'jsoneditor')
pn.extension('tabulator')

class SLEGOApp:
    def __init__(self, config):
        self.config = config
        self.folder_path = config['folder_path']
        self.functionspace = config['functionspace']
        self.dataspace = config['dataspace']
        self.recordspace = config['recordspace']
        self.knowledgespace = config['knowledgespace']
        self.func_file_path = 'func.py'
        self.setup_panel_extensions()
        self.initialize_widgets()
        self.setup_func_module()
        self.setup_event_handlers()
        self.create_layout()

    def setup_panel_extensions(self):
        pn.extension()
        pn.extension(sizing_mode='stretch_both')
        pn.extension('ace', 'jsoneditor')
        pn.extension('tabulator')

    def initialize_widgets(self):
        self.py_files = [file for file in os.listdir(self.functionspace) if file.endswith('.py')]
        self.funcfilecombo = pn.widgets.MultiChoice(
            name='Select Modules',
            value=['util.py', 'func_data_preprocss.py', 'func_yfinance.py', 'llm.py',
                   'func_viz.py', 'func_eda.py', 'func_uci_dataset.py', 'webscrape.py',
                   'func_arxiv.py', 'func_backtest.py', 'func_autogluon.py'],
            options=self.py_files, height=80
        )
        self.compute_btn = pn.widgets.Button(name='Compute', height=50, button_type='primary')
        self.savepipe_btn = pn.widgets.Button(name='Save Pipeline', height=35)
        self.pipeline_text = pn.widgets.TextInput(value='', placeholder='Input Pipeline Name', height=35)
        self.json_toggle = pn.widgets.Toggle(name='Input mode: text or form', height=35, button_type='warning')
        self.json_editor = pn.widgets.JSONEditor(value={}, mode='form')
        self.input_text = pn.widgets.TextAreaInput(value='', placeholder='input the parameters')
        self.progress_text = pn.widgets.TextAreaInput(value='', placeholder='Input your analytics query here', name='User query inputs for recommendation:', height=150)
        self.output_text = pn.widgets.TextAreaInput(value='', placeholder='Results will be shown here', name='System output message:')
        self.recommendation_btn = pn.widgets.Button(name='Get Recommendation', height=35, button_type='success')
        self.recomAPI_text = pn.widgets.TextInput(value='', placeholder='Your AI API key', height=35)
        self.file_text = pn.widgets.TextInput(value='/dataspace', placeholder='Input the file name')
        self.folder_select = pn.widgets.Select(name='Select Folder', options=[item for item in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, item))] + ['/'], value='dataspace', height=50)
        self.filefolder_confirm_btn = pn.widgets.Button(name='Confirm')
        self.file_view = pn.widgets.Button(name='View')
        self.file_download = pn.widgets.Button(name='Download')
        self.file_upload = pn.widgets.Button(name='Upload')
        self.file_input = pn.widgets.FileInput(name='Upload file')
        self.file_delete = pn.widgets.Button(name='Delete')
        self.file_table = self.create_file_table()
        self.widget_tab = pn.Tabs(('json input', self.json_editor), ('text input', self.input_text))

    def setup_func_module(self):
        self.delete_func_file()
        self.create_func_file()
        import func
        importlib.reload(func)
        self.func = func
        self.funccombo = self.create_multi_select_combobox(self.func)

    def delete_func_file(self):
        if os.path.exists(self.func_file_path):
            os.remove(self.func_file_path)

    def create_func_file(self):
        with open(self.func_file_path, 'w') as func_file:
            for py_file in self.funcfilecombo.value:
                file_path = os.path.join(self.functionspace, py_file)
                with open(file_path, 'r') as file:
                    func_file.write(file.read() + '\n')

    def create_file_table(self):
        selected_folder_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'))
        file_list = os.listdir(selected_folder_path)
        df_file = pd.DataFrame(file_list, columns=['Filter Files :'])
        return pn.widgets.Tabulator(df_file, theme='semantic-ui', header_filters=True, layout='fit_data_table', show_index=True, widths={'index': 30}, margin=(0,0,30,0))

    def setup_event_handlers(self):
        self.funcfilecombo.param.watch(self.funcfilecombo_change, 'value')
        self.funccombo.param.watch(self.funccombo_change, 'value')
        self.input_text.param.watch(self.input_text_change, 'value')
        self.json_toggle.param.watch(self.json_toggle_clicked, 'value')
        self.json_editor.param.watch(self.json_editor_change, 'value')
        self.recommendation_btn.param.watch(self.recommendation_btn_clicked, 'value')
        self.compute_btn.on_click(self.compute_btn_clicked)
        self.savepipe_btn.on_click(self.save_pipeline)
        self.filefolder_confirm_btn.on_click(self.on_filefolder_confirm_btn_click)
        self.file_view.on_click(self.on_file_buttons_click)
        self.file_download.on_click(self.on_file_buttons_click)
        self.file_upload.on_click(self.on_file_buttons_click)
        self.file_delete.on_click(self.on_file_buttons_click)
        self.folder_select.param.watch(self.folder_select_changed, 'value')

    def create_layout(self):
        widget_input = pn.Column(pn.layout.Divider(height=10, margin=(5)), self.widget_tab)
        widget_btns = pn.Row(self.savepipe_btn, self.pipeline_text)
        widget_updownload = pn.Column(pn.Row(self.file_view, self.file_download), self.file_input, pn.Row(self.file_upload, self.file_delete), height=150)
        widget_files = pn.Column(self.folder_select, pn.Row(self.file_text, self.filefolder_confirm_btn, height=55), self.file_table, widget_updownload, width=250, margin=(0,20,0,0))
        widget_funcsel = pn.Column(self.funcfilecombo, self.funccombo, self.compute_btn, widget_btns)
        widget_recom = pn.Row(self.recommendation_btn, self.recomAPI_text)
        
        self.app = pn.Row(widget_files, pn.Column(widget_funcsel, widget_input), pn.Column(widget_recom, self.progress_text, pn.layout.Divider(height=10, margin=(10)), self.output_text))

    def funcfilecombo_change(self, event):
        self.setup_func_module()

    def funccombo_change(self, event):
        self.output_text.value = ''
        list_funcs = self.funccombo.value
        list_params = [self.extract_parameter(eval('self.func.' + funcchoice)) for funcchoice in list_funcs]
        funcs_params = dict(zip(list_funcs, list_params))
        formatted_data = json.dumps(funcs_params, indent=5)
        self.json_editor.value = funcs_params
        self.input_text.value = str(formatted_data)
        self.output_text.value = self.get_doc_string(self.input_text.value)

    def input_text_change(self, event):
        text = re.sub(r'\bfalse\b', 'False', self.input_text.value, flags=re.IGNORECASE)
        text = text.replace("'", '"')
        try:
            pipeline_dict = json.loads(text)
            pipeline_dict_json = json.dumps(pipeline_dict, indent=4)
            self.input_text.value = pipeline_dict_json
            self.json_editor.value = json.loads(pipeline_dict_json)
            self.output_text.value += '\n Input changed!'
        except ValueError as e:
            self.output_text.value += f'\n Error parsing input: {e}'

    def json_toggle_clicked(self, event):
        self.widget_tab.active = 1 if event.new else 0

    def json_editor_change(self, event):
        text = str(self.json_editor.value)
        text = re.sub(r'\bfalse\b', 'False', text, flags=re.IGNORECASE)
        text = text.replace("'", '"')
        self.input_text.value = text

    def recommendation_btn_clicked(self, event):
        self.output_text.value = 'Asking AI for recommendation: \n'
        user_pipeline = self.json_editor.value
        user_query = self.progress_text.value
        db_path = f'{self.folder_path}/KB.db'
        openai_api_key = self.recomAPI_text.value

        response_text = rc.pipeline_recommendation(db_path, user_query, user_pipeline, openai_api_key)
        self.output_text.value = response_text
        self.output_text.value += '\n\n=================================\n'
        response_text = rc.pipeline_parameters_recommendation(user_query, response_text, openai_api_key)

        text = str(response_text)
        text = re.sub(r"\b(false|False)\b", '"false"', text, flags=re.IGNORECASE)

        self.output_text.value += response_text

        services = json.loads(response_text)
        keys = list(services.keys())
        self.funccombo.value = keys

        rec_string = json.dumps(text, indent=4)
        self.json_editor.value = rec_string

    def compute_btn_clicked(self, event):
        self.progress_text.value = 'Computing...'
        self.widget_tab.active = 1 - self.widget_tab.active
        time.sleep(1)
        self.widget_tab.active = 1 - self.widget_tab.active

        pipeline_dict = self.json_editor.value
        self.output_text.value = ''

        for function_name, parameters in pipeline_dict.items():
            self.progress_text.value = f'Computing {function_name}...'
            try:
                start_time = time.time()
                function = getattr(self.func, function_name)
                result = function(**parameters)
                result_string = str(result)
                words_iterator = iter(result_string.split())
                first_x_words = itertools.islice(words_iterator, 500)
                compute_time = time.time() - start_time

                self.output_text.value += f"\n===================={function_name}====================\n\n"
                self.output_text.value += f"Function computation Time: {compute_time:.4f} seconds\n\n"
                self.output_text.value += " ".join(first_x_words)
            except Exception as e:
                self.output_text.value += f"\n===================={function_name}====================\n\n"
                self.output_text.value += f"Error occurred: {str(e)}\n"

        self.save_record('recordspace', pipeline_dict)
        self.progress_text.value = 'Done!'
        self.on_filefolder_confirm_btn_click(None)
        self.refresh_file_table()

    def save_pipeline(self, event):
        pipeline_name = self.pipeline_text.value if self.pipeline_text.value else '__'
        text = re.sub(r'\bfalse\b', 'False', self.input_text.value, flags=re.IGNORECASE)
        data = ast.literal_eval(text)
        self.save_record('knowledgespace', data, pipeline_name)
        self.on_filefolder_confirm_btn_click(None)

    def on_file_buttons_click(self, event):
        self.output_text.value = ''
        file_list = self.file_table.selected_dataframe.values.tolist()
        if len(file_list) != 0:
            if event.obj.name == 'View':
                self.output_text.value = ''
                for filename in file_list:
                    self.output_text.value += f"\n\n===================={str(filename)}====================\n\n"
                    file_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'), filename[0])
                    with open(file_path, 'r') as file:
                        self.output_text.value += file.read()
            elif event.obj.name == 'Download':
                self.output_text.value = 'The file is already saved to your folder!'
            elif event.obj.name == 'Upload':
                self.output_text.value = 'Please use the file input widget to upload!'
            elif event.obj.name == 'Delete':
                self.output_text.value = 'Delete functionality is not implemented for safety reasons.'
        else:
            self.output_text.value = 'Please select a file to view, download, upload or delete!'

    def on_filefolder_confirm_btn_click(self, event):
        selected_folder_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'))
        file_list = os.listdir(selected_folder_path)
        df_file = pd.DataFrame(file_list, columns=['Filter Files :'])
        self.file_table.value = df_file

    def folder_select_changed(self, event):
        self.file_text.value = '/' + str(self.folder_select.value)
        self.on_filefolder_confirm_btn_click(None)

    def get_doc_string(self, pipeline):
            text = self.input_text.value
            output = ''
            data = json.loads(text)
            for key in data.keys():
                output += f'#######{str(key)}#######\n'
                try:
                    output += getattr(self.func, key).__doc__ + '\n'
                except AttributeError:
                    output += 'No docstring found for this function\n'
            return output

    @staticmethod
    def is_colab_runtime():
        try:
            import google.colab
            return True
        except ImportError:
            return False

    def save_record(self, space, data, pipeline_name=None):
        if pipeline_name is None:
            filename = datetime.now().strftime("record_%Y%m%d_%H%M%S.json")
        else:
            filename = pipeline_name + '.json'

        full_path = os.path.join(self.config[space], filename)

        with open(full_path, "w") as file:
            json.dump(data, file, indent=5)

        self.refresh_file_table()

    def refresh_file_table(self):
        selected_folder_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'))
        file_list = os.listdir(selected_folder_path)
        df_file = pd.DataFrame(file_list, columns=['Filter Files :'])
        self.file_table.value = df_file

    def create_multi_select_combobox(self, target_module):
        """
        Creates a multi-select combobox with all functions from the target_module.
        """
        module_name = target_module.__name__
        functions = [name for name, obj in inspect.getmembers(target_module, inspect.isfunction)
                     if obj.__module__ == module_name and not name.startswith('_')]
        multi_combobox = pn.widgets.MultiChoice(name='Functions:', options=functions, height=150)
        return multi_combobox

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

    def run(self):
        if not self.is_colab_runtime():
            template = pn.template.MaterialTemplate(
                title='SLEGO - Software Lego: A Collaborative and Modular Architecture for Data Analytics',
                sidebar=[],
            )
            template.main.append(self.app)
            template.show()
        else:
            from IPython.display import display
            display(self.app)

    def test_function(self, input_string:str='Hello!', 
              output_file_path:str='dataspace/output.txt'):
        """
        A simple function to save the provided input string to a specified text file and return the string.

        Parameters:
        - input_string (str): The string to be saved.
        - output_file_path (str): The file path where the string should be saved.

        Returns:
        - str: The same input string.
        """
        with open(output_file_path, 'w') as file:
            file.write(input_string)
        return input_string

    def compute(self, module_name, input):
        module = __import__(module_name)
        pipeline_dict = json.loads(input)
        output = ""
        for function_name, parameters in pipeline_dict.items():
            function = eval(f"module.{function_name}")
            result = function(**parameters)
            output += f"\n===================={function_name}====================\n\n"
            output += str(result)
        return output

    def combine_json_files(self, directory, output_file):
        """
        Combine all JSON files in a directory into a single JSON file.

        Args:
        directory (str): The directory containing JSON files.
        output_file (str): The path to the output JSON file.
        """
        combined_data = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    combined_data.append(data)
        with open(output_file, 'w') as file:
            json.dump(combined_data, file, indent=4)
        print("All JSON files have been combined into:", output_file)

