import os
import requests
import zipfile
import io

def downloadData(dropboxLink, outDirName):
    """
    Downloads a ZIP file from a Dropbox shared link and extracts it into ./data/outDirName
    Only downloads and extracts if the directory does not already exist or is empty.

    Args:
        dropboxLink (str): The Dropbox shared link (to a ZIP or folder).
        outDirName (str): The name of the subfolder in ./data where files will be extracted.
    """
    # Convert Dropbox link to force direct download
    if dropboxLink.endswith('dl=0'):
        download_url = dropboxLink[:-1] + '1'
    elif dropboxLink.endswith('dl=1'):
        download_url = dropboxLink
    else:
        raise ValueError("Invalid Dropbox link. It should end with 'dl=0' or 'dl=1'.")

    output_dir = os.path.join('dataSets', outDirName)
    # Check if directory exists and is not empty
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        print(f"Data already exists in ./{output_dir}, skipping download.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading ZIP from {download_url} ...")
    response = requests.get(download_url)
    response.raise_for_status()  # Raise error if download failed

    print(f"Extracting contents to ./{output_dir} ...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(output_dir)

    print("Download and extraction complete.")


def downloadDataSets():
    print("Downloading datasets...")
    link = "https://www.dropbox.com/scl/fi/gnonzprr3ykf3543hye8l/carData.zip?rlkey=0x120m0pn82futkwbvztvqo10&st=qz1ipf1h&dl=0"
    downloadData(link, "cars") # recordar donde los descargamos para referenciarlos en train_test_split
    # agregar mas datasets

if __name__ == "__main__":
    # test
    downloadDataSets()