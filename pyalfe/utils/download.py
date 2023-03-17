import os

import requests


def download_archive(url: str, download_dir: str, archive_name: str = None):

    response = requests.get(url)
    if not archive_name:
        archive_name = url.split('/')[-1]

    archive_file_path = os.path.join(download_dir, archive_name)
    with open(archive_file_path, 'wb') as file:
        file.write(response.content)

    return archive_file_path
