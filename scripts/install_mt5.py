import os
import requests
from subprocess import Popen

URL = "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"
FILENAME = "mt5setup.exe"

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

def install(filename):
    print(f"Launching installer: {filename}")
    if os.path.exists(filename):
        try:
             # Run asynchronously so script doesn't hang
            Popen([filename], shell=True)
            print("Installer launched. Please complete the installation steps.")
        except Exception as e:
            print(f"Failed to launch installer: {e}")
    else:
        print("File not found.")

if __name__ == "__main__":
    if download_file(URL, FILENAME):
        install(FILENAME)
