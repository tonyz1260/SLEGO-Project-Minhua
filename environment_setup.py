import os
import sys
import subprocess
import platform
from typing import Dict, Any, Optional

def detect_environment() -> str:
    """Detect the current runtime environment."""
    print("Detecting environment...")
    if 'google.colab' in sys.modules:
        print("Environment detected: Google Colab")
        return 'colab'
    elif os.environ.get('CODESPACES', 'false').lower() == 'true':
        print("Environment detected: GitHub Codespaces")
        return 'github-codespaces'
    else:
        print("Environment detected: Local Jupyter")
        return 'local-jupyter'

def get_environment_config() -> Dict[str, Any]:
    """Get the configuration based on the current runtime environment."""
    print("Getting environment configuration...")
    config = {}
    env = detect_environment()

    if env == 'colab':
        print("Setting up configuration for Google Colab environment.")
        config['drive_mainfolder'] = '/content/drive/MyDrive/SLEGO'
        config['drive_folder'] = '/content/drive/MyDrive/'
        # Mount Google Drive
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=True)
    elif env == 'github-codespaces':
        print("Setting up configuration for GitHub Codespaces environment.")
        config['drive_mainfolder'] = '/workspaces/SLEGO-Project'
        config['drive_folder'] = '/workspaces/'
    else:  # local-jupyter
        print("Setting up configuration for Local Jupyter environment.")
        # Adjusted to use the user's home directory
        home_dir = os.path.expanduser('~')
        config['drive_mainfolder'] = os.path.join(home_dir, 'SLEGO')
        config['drive_folder'] = home_dir

    config['repo_url'] = 'https://github.com/alanntl/SLEGO-Project.git'
    config['slego_env'] = os.path.join(config['drive_folder'], 'slego_env_v0_0_1')
    config['requirements_file'] = os.path.join(config['drive_mainfolder'], 'requirements.txt')

    # Set up workspace folders
    config['folder_path'] = os.path.join(config['drive_mainfolder'], 'slegospace')
    config['dataspace'] = os.path.join(config['folder_path'], 'dataspace')
    config['recordspace'] = os.path.join(config['folder_path'], 'recordspace')
    config['functionspace'] = os.path.join(config['folder_path'], 'functionspace')
    config['knowledgespace'] = os.path.join(config['folder_path'], 'knowledgespace')
    config['ontologyspace'] = os.path.join(config['folder_path'], 'ontologyspace')

    os.environ['DRIVE_MAINFOLDER'] = config['drive_mainfolder']
    os.environ['DRIVE_FOLDER'] = config['drive_folder']

    print("Configuration settings:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    return config

def run_command(command: list, check: bool = True, **kwargs) -> Optional[bool]:
    """Run a subprocess command with error handling."""
    try:
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=check, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(command)}: {e}")
        return None
    return True

def setup_repository(config: Dict[str, Any]):
    """Set up or update the repository based on the current environment."""
    print("Setting up repository...")
    repo_path = config['drive_mainfolder']
    repo_url = config['repo_url']
    original_dir = os.getcwd()
    print(f"Current working directory: {original_dir}")
    print(f"Repository path: {repo_path}")
    print(f"Repository URL: {repo_url}")

    if not os.path.exists(repo_path):
        print(f"Repository path does not exist. Cloning repository to {repo_path}")
        os.makedirs(repo_path, exist_ok=True)
        os.chdir(os.path.dirname(repo_path))
        if run_command(['git', 'clone', repo_url, os.path.basename(repo_path)]):
            print("Repository cloned successfully.")
        else:
            print("Failed to clone repository.")
    else:
        print("Repository path exists.")
        os.chdir(repo_path)
        if os.path.exists(os.path.join(repo_path, '.git')):
            print("Git repository found.")
            if run_command(['git', 'fetch']):
                print("Fetched latest changes from remote.")
                result = subprocess.run(['git', 'status'], capture_output=True, text=True)
                if 'Your branch is behind' in result.stdout:
                    print("Repository is behind. Pulling updates...")
                    if run_command(['git', 'pull']):
                        print("Repository updated.")
                    else:
                        print("Failed to pull updates.")
                else:
                    print("Repository is up to date.")
            else:
                print("Failed to fetch from remote.")
        else:
            print(f"Warning: {repo_path} exists but is not a git repository.")

    os.chdir(original_dir)  # Return to the original directory
    print(f"Returned to original directory: {original_dir}")

def setup_virtual_environment(config: Dict[str, Any]):
    """Set up the virtual environment and install requirements."""
    print("Setting up virtual environment...")
    env = detect_environment()
    slego_env = config['slego_env']
    requirements_file = config['requirements_file']

    python_executable = sys.executable  # Path to the Python executable
    print(f"Using Python executable: {python_executable}")

    if not os.path.exists(slego_env):
        print(f"Virtual environment not found at {slego_env}. Creating new virtual environment.")
        run_command([python_executable, '-m', 'venv', slego_env])
    else:
        print(f"Virtual environment found at {slego_env}.")

    activate_script = ''
    if platform.system() == 'Windows':
        activate_script = os.path.join(slego_env, 'Scripts', 'activate_this.py')
    else:
        activate_script = os.path.join(slego_env, 'bin', 'activate_this.py')

    if os.path.exists(activate_script):
        print(f"Activating virtual environment from {activate_script}")
        with open(activate_script) as f:
            exec(f.read(), {'__file__': activate_script})
    else:
        print(f"Activation script not found at {activate_script}")

    # Install requirements
    if os.path.exists(requirements_file):
        print(f"Installing requirements from {requirements_file}")
        run_command([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        run_command([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
    else:
        print(f"Warning: Requirements file not found at {requirements_file}")
        print("Skipping requirements installation.")

def setup_workspace(config: Dict[str, Any]):
    """Set up the workspace folders and change the working directory."""
    print("Setting up workspace folders...")
    workspace_folders = [config['folder_path'], config['dataspace'], config['recordspace'],
                         config['functionspace'], config['knowledgespace'], config['ontologyspace']]
    for folder in workspace_folders:
        if not os.path.exists(folder):
            print(f"Creating folder: {folder}")
            os.makedirs(folder, exist_ok=True)
        else:
            print(f"Folder already exists: {folder}")

    os.chdir(config['folder_path'])
    print(f"Working directory changed to: {os.getcwd()}")

    if detect_environment() == 'colab':
        from google.colab import files
        print(f"Viewing folder in Colab: {config['folder_path']}")
        files.view(config['folder_path'])

def setup_environment():
    """Set up the environment, repository, virtual environment, and workspace."""
    print("Starting environment setup...")
    config = get_environment_config()
    setup_repository(config)
    setup_virtual_environment(config)
    setup_workspace(config)
    print("Environment setup completed.")
    return config

# Global variable to store the configuration
global_config = None

# Automatically run setup when the module is imported
global_config = setup_environment()
