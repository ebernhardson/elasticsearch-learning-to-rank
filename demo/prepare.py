from urllib.parse import urljoin
import requests


# Use this to download the library taking the version from the configuration file
from log_conf import Logger
from utils import RANKLIB_JAR


def download_ltr_resource(resource):
    """
    Downloads the provided resource from the o19s website to your local folder. When running this script, the defaults
    for the demo are downloaded. You can also used it yourself.
    :param resource: the name of the resource to download
    :return:
    """
    ltr_domain = 'https://people.wikimedia.org/~ebernhardson/esltr/'
    resource_url = urljoin(ltr_domain, resource)
    with open(resource, 'wb') as dest:
        Logger.logger.info("GET %s" % resource_url)
        resp = requests.get(resource_url, stream=True)
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                dest.write(chunk)


if __name__ == "__main__":
    download_ltr_resource('discernatron.json')
