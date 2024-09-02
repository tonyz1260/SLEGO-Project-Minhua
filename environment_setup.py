import os
import sys
import subprocess
import platform
from typing import Dict, Any, Optional

def detect_environment() -> str:
    """Detect the current runtime environment."""
    if 'google.colab' in sys.modules:
        return 'colab'
    elif 'CODESPACES' in os.environ and os.environ['CODESPACES'] == 'true':
        return 'github-codespaces'
    else:
        return 'local-jupyter'

def get_environment_config() -> Dict[str, Any]:
    """Get the configuration based on the current runtime environment."""
    config = {}
    env = detect_environment()
    
    if env == 'colab':
        print("Running in Google Colab environment.")
        config['drive_mainfolder'] = '/content/drive/MyDrive/SLEGO'
        config['drive_folder'] = '/content/drive/MyDrive/'
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
    elif env == 'github-codespaces':
        print("Running in GitHub Codespaces environment. Let's go hhhhhhhhahah")
        config['drive_mainfolder'] = '/workspaces/SLEGO-Project-Minhua'
        config['drive_folder'] = '/workspaces/'
    else:  # local-jupyter
        print("Running in a local Jupyter environment.")
        gmailaccount = os.environ.get('GMAIL_ACCOUNT', 'default@gmail.com')
        config['drive_mainfolder'] = f"/Users/an/Library/CloudStorage/GoogleDrive-{gmailaccount}/My Drive/SLEGO"
        config['drive_folder'] = f"/Users/an/Library/CloudStorage/GoogleDrive-{gmailaccount}/My Drive/"
    
    config['repo_url'] = 'https://github.com/alanntl/SLEGO-Project.git'
    config['slego_env'] = f"{config['drive_folder']}/slego_env_v0_0_1"
    config['requirements_file'] = f"{config['drive_mainfolder']}/requirements.txt"
    
    # Set up workspace folders
    config['folder_path'] = f"{config['drive_mainfolder']}/slegospace"
    config['dataspace'] = f"{config['folder_path']}/dataspace"
    config['recordspace'] = f"{config['folder_path']}/recordspace"
    config['functionspace'] = f"{config['folder_path']}/functionspace"
    config['knowledgespace'] = f"{config['folder_path']}/knowledgespace"
    config['ontologyspace'] = f"{config['folder_path']}/ontologyspace"
    
    os.environ['DRIVE_MAINFOLDER'] = config['drive_mainfolder']
    os.environ['DRIVE_FOLDER'] = config['drive_folder']
    
    return config

def run_command(command: list, check: bool = True, **kwargs) -> Optional[bool]:
    """Run a subprocess command with error handling."""
    try:
        subprocess.run(command, check=check, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(command)}: {e}")
        return None
    return True

def setup_repository(config: Dict[str, Any]):
    """Set up or update the repository based on the current environment."""
    repo_path = config['drive_mainfolder']
    repo_url = config['repo_url']
    original_dir = os.getcwd()

    if not os.path.exists(repo_path):
        os.makedirs(repo_path, exist_ok=True)
        os.chdir(repo_path)
        if run_command(['git', 'init']):
            if run_command(['git', 'remote', 'add', 'origin', repo_url]):
                if run_command(['git', 'fetch']):
                    if run_command(['git', 'checkout', '-b', 'master', '--track', 'origin/master']):
                        print("Repository initialized and master branch checked out.")
                    else:
                        print("Failed to checkout master branch.")
                else:
                    print("Failed to fetch from remote.")
            else:
                print("Failed to add remote.")
        else:
            print("Failed to initialize git repository.")
    else:
        os.chdir(repo_path)
        if os.path.exists(os.path.join(repo_path, '.git')):
            if run_command(['git', 'fetch']):
                result = subprocess.run(['git', 'diff', '--name-only', 'HEAD', 'origin/master'], capture_output=True, text=True)
                changed_files = result.stdout.splitlines()
                if any(changed_files):
                    if run_command(['git', 'pull']):
                        print("Repository updated.")
                else:
                    print("No changes detected; no update necessary.")
            else:
                print("Failed to fetch from remote.")
        else:
            print(f"Warning: {repo_path} exists but is not a git repository.")

    os.chdir(original_dir)  # Return to the original directory

def setup_virtual_environment(config: Dict[str, Any]):
    """Set up the virtual environment and install requirements."""
    env = detect_environment()
    slego_env = config['slego_env']
    requirements_file = config['requirements_file']

    if not os.path.exists(slego_env):
        run_command([sys.executable, '-m', 'pip', 'install', 'virtualenv'])
        run_command([sys.executable, '-m', 'virtualenv', slego_env])

    python_version = f"python{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}"
    
    if env == 'colab':
        activate_this = f"{slego_env}/bin/activate_this.py"
        exec(open(activate_this).read(), {'__file__': activate_this})
        sys.path.append(f"{slego_env}/lib/{python_version}/site-packages")
    elif env == 'github-codespaces':
        os.environ['VIRTUAL_ENV'] = slego_env
        os.environ['PATH'] = f"{slego_env}/bin:{os.environ['PATH']}"
    else:  # local-jupyter
        activate_this = f"{slego_env}/bin/activate_this.py"
        exec(open(activate_this).read(), {'__file__': activate_this})

    if os.path.exists(requirements_file):
        run_command([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
    else:
        print(f"Warning: Requirements file not found at {requirements_file}")
        print("Skipping requirements installation.")

def setup_workspace(config: Dict[str, Any]):
    """Set up the workspace folders and change the working directory."""
    for folder in [config['folder_path'], config['dataspace'], config['recordspace'], 
                   config['functionspace'], config['knowledgespace']]:
        os.makedirs(folder, exist_ok=True)
    
    os.chdir(config['folder_path'])
    print(f"Working directory changed to: {os.getcwd()}")

    if detect_environment() == 'colab':
        from google.colab import files
        files.view(config['folder_path'])

def setup_environment():
    """Set up the environment, repository, virtual environment, and workspace."""
    config = get_environment_config()
    setup_repository(config)
    setup_virtual_environment(config)
    setup_workspace(config)
    return config

# Global variable to store the configuration
global_config = None

# Automatically run setup when the module is imported
global_config = setup_environment()