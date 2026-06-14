"""
A library for easily accessing the Shadertoy API.
For more info about the Shadertoy API, refer to shadertoy.com/howto
Note: only shaders listed as Public+API are available through the Shadertoy API.
"""

from io import BytesIO
import json

# shadertoy.com sits behind Cloudflare bot protection that blocks clients by
# TLS fingerprint, so plain `requests` gets a 403 regardless of headers.
# curl_cffi impersonates a real Chrome handshake and passes without cookies.
try:
    from curl_cffi import requests
    _request_kwargs = {"impersonate": "chrome"}
except ImportError:
    import requests
    _request_kwargs = {}

base_url = "https://www.shadertoy.com"
api_base_url = base_url + "/api/v1/shaders"

classifiers = ("name", "love", "popular", "newest", "hot")
filters = ("vr", "soundoutput", "soundinput", "webcam", "multipass", "musicstream")

class ShadertoyAPIError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class App(object):
    """ A class for accessing the Shadertoy API. """
    def __init__(self, key, user_agent="python-application"):
        self.key = key
        # When impersonating Chrome, the User-Agent must stay consistent with
        # the TLS fingerprint or Cloudflare rejects the request anyway.
        self.headers = {} if _request_kwargs else {"User-Agent": user_agent}
        super().__init__()

    def _get_json(self, url):
        response = requests.get(url, headers=self.headers, **_request_kwargs)
        if response.status_code != 200:
            if response.status_code == 403 and not _request_kwargs:
                raise ShadertoyAPIError(
                    "Blocked by Cloudflare bot protection (HTTP 403). "
                    "Install the 'curl_cffi' package into Houdini's Python to fix this, "
                    "or import the shader from a JSON file instead."
                )
            raise ShadertoyAPIError(f"Shadertoy API request failed (HTTP {response.status_code})")

        parsed_json = json.loads(response.content.decode("utf-8"))
        if "Error" in parsed_json: # The Shadertoy API returned an error message
            error = parsed_json["Error"]
            # "Shader not found" is also returned when a shader exists but the author
            # didn't publish it as "Public + API", which is easy to misdiagnose.
            if "not found" in error.lower():
                error = (
                    "Shader not found, or not published with 'Public + API' access. "
                    "Open it on shadertoy.com and check the shader's visibility setting."
                )
            raise ShadertoyAPIError(error) # Show the error message

        else:
            return parsed_json

    def download_media_file(self, path):
        """
        Downloads a shadertoy media file from the given path (relative to shadertoy.com)
        and returns it as a file-like io.BytesIO object.
        """

        url = base_url + path
        response = requests.get(url, headers=self.headers, **_request_kwargs)
        if response.status_code != 200: # Did not get the desired response, probably 404 file not found
            raise ShadertoyAPIError(f"Failed to download media file {path} (HTTP {response.status_code})")

        else:
            return BytesIO(response.content) # Return a file-like object for accessing the media file

    def download_shader_icon(self, shader_id):
        """
        Downloads the icon for the shader with the given id and returns it as a
        file-like io.BytesIO object (note that shader icons are in JPEG format).
        """

        url = base_url + "/media/shaders/" + shader_id + ".jpg"
        response = requests.get(url, headers=self.headers, **_request_kwargs)
        if response.status_code != 200: # Did not get the desired response, probably 404 file not found
            raise ShadertoyAPIError(f"Failed to download icon for shader {shader_id} (HTTP {response.status_code})")

        else:
            return BytesIO(response.content) # Return a file-like object for accessing the image file

    def get_embeddable_url(self, shader_id, enable_gui=True, start_time=10, paused=True, muted=False):
        """
        Returns an embeddable URL for the shader with the given id, and configures
        it with the given initial settings.
        """

        url = base_url + "/embed/" + shader_id + "?"
        url += "gui=" + str(enable_gui).lower() + "&"
        url += "t=" + str(start_time) + "&"
        url += "paused=" + str(paused).lower() + "&"
        url += "muted=" + str(muted).lower()
        return url

    def query(self, keywords=[], sort_by=None, filter=None, start_index=0, num_shaders="all"):
        """
        Queries the shadertoy database for shaders matching the given filter
        and returns a list with the given number of their IDs, starting at the
        given index and sorted by the given classifier.

        Classifiers: "name", "love", "popular", "newest", "hot"
        Filters: "vr", "soundoutput", "soundinput", "webcam", "multipass", "musicstream"

        All classifiers and filters can be accessed from shadertoy.classifiers and shadertoy.filters
        """

        # https://www.shadertoy.com/api/v1/shaders/query/string?key=appkey
        url = api_base_url + "/query/"
        for keyword in keywords:
            url += keyword + "+"

        url = url[:-1] + "?"
        if sort_by is not None:
            url += "sort=" + sort_by + "&"

        if filter is not None:
            url += "filter=" + filter + "&"

        if start_index > 0:
            url += "from=" + str(start_index) + "&"

        if num_shaders != "all":
            url += "num=" + str(num_shaders) + "&"

        url += "key=" + self.key
        response_data = self._get_json(url)
        if response_data["Shaders"] == 0: # No results
            return []

        else:
            return response_data["Results"]

    def get_shader(self, shader_id):
        """ Returns a dictionary containing data about the shader with the given ID. """
        # https://www.shadertoy.com/api/v1/shaders/shaderID?key=appkey
        url = api_base_url + "/" + shader_id + "?key=" + self.key
        return self._get_json(url)["Shader"]

    def get_all_shaders(self):
        """ Returns a list of the IDs of all available shaders. """
        # https://www.shadertoy.com/api/v1/shaders?key=appkey
        url = api_base_url + "?key=" + self.key
        return self._get_json(url)["Results"]
