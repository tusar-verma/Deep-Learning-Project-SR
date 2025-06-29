import os
import requests
import zipfile
import io


###### todavia no recorte el dataset de gatos y perros para que sean de 300-ish imagenes #####

dict_test = {
    "cars": "https://www.dropbox.com/scl/fi/kf6lxlu6ayu30v5wz77cc/cars_test.zip?rlkey=fxpr0vawbzhgd0r5u5r4fwfv8&st=sqqgpu8j&dl=0",
    "cats": "https://www.dropbox.com/scl/fi/i09lf4c4849yxfsett3su/cats_test.zip?rlkey=hpwbi4qcbc0s5uu5jyyasma0b&st=so8h59x7&dl=0",
    "dogs": "https://www.dropbox.com/scl/fi/5axm0tqvxcuaxyg7zxhcf/dogs_test.zip?rlkey=51fwuosdslkxibtdozjscmc1s&st=kkhz53n2&dl=0",
    "landscapes": "https://www.dropbox.com/scl/fi/e8hswg4uz2ibo5lkr2wic/landscape_test.zip?rlkey=rjles5o0d4ar33i7nx2vjvezy&st=23mryl5r&dl=0",
    "people" : "https://www.dropbox.com/scl/fi/5xhhvsp2iwp4vikqg7scm/people_validate.zip?rlkey=t2bt5aqupz6u6z43t23arjjkj&st=lxv79tgc&dl=0"
}

dict_source = {
    "bsd300": "https://www.dropbox.com/scl/fi/kxvxne6w3maal50mh6na3/BSD300.zip?rlkey=bnul455xvn1ylnpr2aes10xok&st=vojdg78q&dl=0",
    "cars": "https://www.dropbox.com/scl/fi/iah6sccsrpetgq5tbp0b0/cars.zip?rlkey=gchvhhp02pvnj8o0256mppckg&st=im29ir3k&dl=0",
    "cats": "https://www.dropbox.com/scl/fi/udyzf36km40na6nejuni6/cat.zip?rlkey=nwkexq48h5e83viw3jyxf88k7&st=tjy3s20e&dl=0",
    "dogs": "https://www.dropbox.com/scl/fi/9cauhbnaih8i695v6p2md/dogs.zip?rlkey=lcdyc48mpk11ubggfqs9mvxlc&st=hotllzyy&dl=0",
    "landscapes": "https://www.dropbox.com/scl/fi/lqwuy6l4e5weui5g87j42/landscape.zip?rlkey=fwgp5ctq4ygz3q5ni7t5g66wq&st=sovhe8sk&dl=0",
    "people" : "https://www.dropbox.com/scl/fi/tvljbws41feghltb064wf/people.zip?rlkey=25u0j55sgsyhxwkcwoug01nf5&st=js8dae4d&dl=0"
}

def downloadData(dropboxLink, output_dir):
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

def downloadTestData(testName):
    """
    Downloads a specific test dataset based on its name.
    Currently supports bsd300, cars, cats, dogs, landscapes, people dataset.
    
    Args:
        testName (str): The name of the test dataset to download.
    """
    
    if testName in dict_test:
        downloadData(dict_test[testName], os.path.join('testSets', testName))
    else:
        raise ValueError(f"Test dataset '{testName}' is not recognized. Available datasets: {list(dict_test.keys())}")


def downloadDataSet(datasetName):
    """
    Downloads a specific dataset based on its name.
    Currently supports cars, cats, dogs dataset.
    
    Args:
        datasetName (str): The name of the dataset to download.
    """
   
    if datasetName in dict_source:
        downloadData(dict_source[datasetName], os.path.join('dataSets', datasetName))
    else:
        raise ValueError(f"Dataset '{datasetName}' is not recognized. Available datasets: {list(dict_source.keys())}")

if __name__ == "__main__":

    for dataset in dict_source.keys():
        downloadDataSet(dataset)

    for testset in dict_test.keys():
        downloadTestData(testset)
