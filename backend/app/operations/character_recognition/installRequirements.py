import subprocess
import pkg_resources

def installRequirements(requirements_file='./requirements.txt'):
    try:
        with open(requirements_file) as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Error: The file '{requirements_file}' was not found.")
        return
    except Exception as e:
        print(f"Error reading the requirements file: {e}")
        return

    to_install = []
    for pkg in packages:
        # Handle different version specifiers
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
        except pkg_resources.DistributionNotFound:
            to_install.append(pkg)
        except pkg_resources.VersionConflict:
            to_install.append(pkg)
        except Exception as e:
            print(f"Error checking package '{pkg_name}': {e}")

    if to_install:
        try:
            subprocess.check_call(['pip', 'install'] + to_install)
            print(f"Installed: {', '.join(to_install)}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install packages: {e}")
        except Exception as e:
            print(f"Unexpected error during installation: {e}")
    else:
        print("All packages are already installed.")
