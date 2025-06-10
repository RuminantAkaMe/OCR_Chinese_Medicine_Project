import subprocess
import pkg_resources
import sys

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
            # Use --user flag if not running as admin/root
            user_flag = []
            if not hasattr(sys, "real_prefix") and not (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
                # Not in a virtualenv, check for admin rights
                try:
                    import ctypes
                    is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
                except Exception:
                    is_admin = False
                if not is_admin:
                    user_flag = ['--user']
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + user_flag + to_install)
            print(f"Installed: {', '.join(to_install)}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install packages: {e}")
        except Exception as e:
            print(f"Unexpected error during installation: {e}")
    else:
        print("All packages are already installed.")
