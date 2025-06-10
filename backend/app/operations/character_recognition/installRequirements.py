import subprocess
import pkg_resources
import sys
import ctypes

def read_requirements(requirements_file='./requirements.txt'):
    """Read the requirements file and return a list of package strings."""
    try:
        with open(requirements_file) as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return packages
    except FileNotFoundError:
        print(f"Error: The file '{requirements_file}' was not found.")
    except Exception as e:
        print(f"Error reading the requirements file: {e}")
    return []

def get_install_list(packages):
    """
    Given a list of package strings, return a list of packages to install by checking
    their current installation status and version conflicts.
    """
    to_install = []
    for pkg in packages:
        # Extract package name (without version specifiers)
        if '==' in pkg:
            pkg_name = pkg.split('==')[0]
        elif '>=' in pkg:
            pkg_name = pkg.split('>=')[0]
        elif '<=' in pkg:
            pkg_name = pkg.split('<=')[0]
        elif '=' in pkg:
            pkg_name = pkg.split('=')[0]
        else:
            pkg_name = pkg

        try:
            pkg_resources.require(pkg)
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            to_install.append(pkg)
        except Exception as e:
            print(f"Error checking package '{pkg_name}': {e}")
    return to_install

def is_running_as_admin():
    """Check if the current process has administrative privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False

def install_packages(packages):
    """Install the list of packages using pip. Use the --user flag if not running as administrator."""
    if not packages:
        print("All packages are already installed.")
        return

    user_flag = []
    # If not in a virtualenv, check for admin rights
    if not hasattr(sys, "real_prefix") and not (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        if not is_running_as_admin():
            user_flag = ['--user']
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + user_flag + packages)
        print(f"Installed: {', '.join(packages)}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
    except Exception as e:
        print(f"Unexpected error during installation: {e}")

def installRequirements(requirements_file='./requirements.txt'):
    """Main function to install required packages from a requirements file."""
    packages = read_requirements(requirements_file)
    if not packages:
        return
    to_install = get_install_list(packages)
    install_packages(to_install)

