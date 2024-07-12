import requests
import tarfile
import os

url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
archive_path = "file.tar"

if __name__ == "__main__":
    response = requests.get(url)
    if response.status_code != 200:
        raise ConnectionRefusedError("Server error")
    with open(archive_path, 'wb') as file:
        file.write(response.content)

    with tarfile.open(archive_path) as tar:
        tar.extractall()

    os.remove(archive_path)  # to reduce the size
