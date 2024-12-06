import io
import json
import logging
import zipfile
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import parse_qsl, urlparse

import requests

from . import __version__
from .core import SYSTEM_MODELS, InterfaceBase, fresh_login

DATA_DIR = Path(__file__).parent.resolve() / "data"

logger = logging.getLogger("ecoinvent_interface")


@lru_cache(maxsize=4)
def get_cached_mapping(version: str, system_model: str) -> dict:
    zf = zipfile.ZipFile(DATA_DIR / "mappings.zip")
    try:
        catalogue = {
            (o["version"], o["system_model"]): o
            for o in json.load(
                io.TextIOWrapper(zf.open("catalogue.json"), encoding="utf-8")
            )
        }
        return json.load(
            io.TextIOWrapper(
                zf.open(catalogue[(version, system_model)]["filename"]),
                encoding="utf-8",
            )
        )
    except KeyError:
        raise KeyError(f"Combination {version} + {system_model} not yet cached")


class MissingProcess(BaseException):
    """Operation not possible because no process selected"""

    pass


def selected_process(f):
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "dataset_id"):
            raise MissingProcess("Must call `.select_process()` first")
        return f(self, *args, **kwargs)

    return wrapper


def split_url(url: str) -> Tuple[str, dict]:
    """Split a URL with params into a base path and a params dict"""
    nt = urlparse(url)
    return nt.path, dict(parse_qsl(nt.query))


class ProcessFileType(Enum):
    upr = "Unit Process"
    lci = "Life Cycle Inventory"
    lcia = "Life Cycle Impact Assessment"
    pdf = "Dataset Report"
    undefined = "Undefined (unlinked and multi-output) Dataset Report"


ZIPPED_FILE_TYPES = (ProcessFileType.lci, ProcessFileType.lcia, ProcessFileType.upr)


def as_tuple(version_string: str) -> Tuple[int, int]:
    return tuple([int(x) for x in version_string.split(".")])


class EcoinventProcess(InterfaceBase):
    def set_release(self, version: str, system_model: str) -> None:
        if version not in self.list_versions():
            raise ValueError(f"Given version {version} not found")
        self.version = version

        system_model = SYSTEM_MODELS.get(system_model, system_model)
        if system_model == "undefined" and as_tuple(version) >= (3, 10):
            pass
        elif system_model not in self.list_system_models(self.version):
            raise ValueError(
                f"Given system model '{system_model}' not available in {version}"  # NOQA E713
            )
        self.system_model = system_model

    def select_process(
        self,
        attributes: Optional[dict] = None,
        filename: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> None:
        if not hasattr(self, "system_model"):
            raise ValueError("Must call `.set_release()` first")

        if dataset_id:
            self.dataset_id = dataset_id
        elif filename:
            mapping = {
                obj["filename"]: obj["index"]
                for obj in get_cached_mapping(
                    version=self.version, system_model=self.system_model
                )
            }
            try:
                self.dataset_id = mapping[filename]
            except KeyError:
                raise KeyError(f"Can't find filename `{filename}` in mapping data")
        elif attributes:
            label_mapping = {
                "reference product": "reference_product",
                "name": "activity_name",
                "location": "geography",
            }
            valid_keys = set(label_mapping).union(set(label_mapping.values()))
            mapped_attributes = {
                label_mapping.get(key, key): value
                for key, value in attributes.items()
                if key in valid_keys
            }
            possibles = [
                obj
                for obj in get_cached_mapping(
                    version=self.version, system_model=self.system_model
                )
                if all(
                    obj.get(key) == value for key, value in mapped_attributes.items()
                )
            ]
            if not possibles:
                raise KeyError("Can't find a dataset for these attributes")
            elif len(possibles) > 1:
                raise KeyError(
                    "These attributes don't uniquely identify one dataset - "
                    + f"{len(possibles)} found"
                )
            else:
                self.dataset_id = possibles[0]["index"]
        else:
            raise ValueError(
                "Must give either `attributes`, `filename`, or `integer` to "
                + "choose a process."
            )

    @selected_process
    @fresh_login
    def _json_request(
        self, 
        url: str, 
        additional_params: Optional[dict] = None,
        method: str = 'GET',
        json_data: Optional[dict] = None
    ) -> Union[dict, list]:
        """Make a JSON request to the API.
        
        Args:
            url: The API endpoint URL
            additional_params: Additional query parameters (optional)
            method: HTTP method ('GET' or 'POST')
            json_data: JSON data for POST requests (optional)
            
        Returns:
            Union[dict, list]: The JSON response from the API
            
        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "ecoinvent-api-client-library": "ecoinvent_interface",
            "ecoinvent-api-client-library-version": __version__,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        headers.update(self.custom_headers)

        message = f"""Requesting URL.
        URL: {url}
        Method: {method}
        Class: {self.__class__.__name__}
        Instance ID: {id(self)}
        Version: {__version__}
        User: {self.username}
        """
        logger.debug(message)

        # Only include these params for GET requests or if we're not doing a search
        if method == 'GET' or 'search' not in url:
            params = {
                "dataset_id": self.dataset_id,
                "version": self.version,
                "system_model": self.system_model,
            }
            if additional_params:
                params.update(additional_params)
        else:
            params = additional_params or {}

        if method == 'GET':
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=20,
            )
        else:  # POST
            response = requests.post(
                url,
                params=params,
                headers=headers,
                json=json_data,
                timeout=20,
            )

        # Check for errors
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {str(e)}")
            logger.error(f"Response content: {response.text}")
            raise

        try:
            return response.json()
        except ValueError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Response content: {response.text}")
            raise
    def get_basic_info(self) -> dict:
        return self._json_request(self.urls["api"] + "spold")
    @selected_process
    def get_documentation(self) -> dict:
        return self._json_request(self.urls["api"] + "spold/documentation")
    @selected_process
    def get_exchanges(self) -> dict:
        return self._json_request(self.urls['api'] + 'spold/exchanges')
    @selected_process
    def get_methods(self):
        return self._json_request(self.urls['api'] + 'spold/methods')
    @selected_process
    def get_related_datasets(self):
        return self._json_request(self.urls['api'] + 'spold/related_datasets')
    @selected_process
    def get_consuming_activities(self):
        return self._json_request(self.urls['api'] + 'spold/consuming_activities')
    @selected_process
    def get_lci(self):
        return self._json_request(self.urls['api'] + 'spold/lci_results')
    @selected_process
    def get_direct_contributions(self, indicator_id: str) -> dict:
        """Get direct contributions for a specific impact indicator.
        
        This method retrieves the direct contributions data for a specific impact assessment
        indicator for the currently selected process.
        
        Args:
            indicator_id: The unique identifier of the impact assessment indicator
            
        Returns:
            dict: A dictionary containing the direct contributions data
            
        Raises:
            MissingProcess: If no process is selected
            requests.exceptions.RequestException: If the API request fails
        """
        return self._json_request(
            self.urls['api'] + 'spold/direct_contributions',
            additional_params={'indicator_id': indicator_id}
        )
    
    
    def get_impacts(self, method_id: str) -> dict:
        """Get LCIA results for a specific impact assessment method.
        
        Args:
            method_id: The ID of the impact assessment method
            
        Returns:
            dict: The LCIA results for the specified method
        """
        return self._json_request(
            self.urls['api'] + 'spold/lcia_results',
            additional_params={'method_id': method_id}
        )
    @fresh_login  # Remove @selected_process as search doesn't need a selected process
    def searcher(
        self,
        query: Optional[str] = None,
        current_page: int = 1,
        page_size: int = 5,
        geography: Optional[list] = None,
        isic_section: Optional[list] = None,
        isic_class: Optional[list] = None,
        activity_type: Optional[list] = None,
        sector: Optional[list] = None,
    ) -> dict:
        """Search datasets for the current version and system model.
        
        Args:
            query: Search term (optional)
            current_page: Page number (default: 1)
            page_size: Number of results per page (default: 5)
            geography: List of geography filters
            isic_section: List of ISIC section filters
            isic_class: List of ISIC class filters
            activity_type: List of activity type filters
            sector: List of sector filters
            
        Returns:
            dict: The search results including pagination info and filtered datasets
        """
        if not hasattr(self, "version") or not hasattr(self, "system_model"):
            raise ValueError("Must call set_release() first")
                
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "ecoinvent-api-client-library": "ecoinvent_interface",
            "ecoinvent-api-client-library-version": __version__,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        headers.update(self.custom_headers)

        # Build search request body
        request_body = {
            "filters": {},
            "query": query or "",
            "currentPage": current_page,
            "pageSize": page_size
        }
        
        # Add optional filters
        if any([geography, isic_section, isic_class, activity_type, sector]):
            filters = {}
            if geography:
                filters["geography"] = geography
            if isic_section:
                filters["isic_section"] = isic_section
            if isic_class:
                filters["isic_class"] = isic_class
            if activity_type:
                filters["activity_type"] = activity_type
            if sector:
                filters["sector"] = sector
            request_body["filters"] = filters

        # Make the request
        url = f"{self.urls['api']}search/{self.version}/{self.system_model}"
        
        response = requests.post(
            url,
            headers=headers,
            json=request_body,
            timeout=20
        )

        # Check for errors
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {str(e)}")
            logger.error(f"Response content: {response.text}")
            raise

        try:
            return response.json()
        except ValueError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Response content: {response.text}")
            raise
    def get_file(self, file_type: ProcessFileType, directory: Path) -> Path:
        files = {
            obj.pop("name"): obj
            for obj in self._json_request(self.urls["api"] + "spold/export_file_list")
        }
        try:
            meta = files[file_type.value]
        except KeyError:
            available = list(files)
            raise KeyError(f"Can't find {file_type} in available options: {available}")

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "ecoinvent-api-client-library": "ecoinvent_interface",
            "ecoinvent-api-client-library-version": __version__,
        }
        headers.update(self.custom_headers)
        if meta.get("type").lower() == "xml":
            headers["Accept"] = "text/plain"

        url, params = split_url(meta["url"])
        suffix = meta["type"].lower()
        filename = (
            f"ecoinvent-{self.version}-{self.system_model}-{file_type.name}-"
            + f"{self.dataset_id}.{suffix}"
        )

        if file_type == ProcessFileType.undefined:
            s3_link = requests.get(
                self.urls["api"][:-1] + url, params=params, headers=headers, timeout=20
            ).json()["download_url"]
            self._streaming_download(
                url=s3_link, params={}, directory=directory, filename=filename
            )
            return directory / filename

        self._streaming_download(
            url=self.urls["api"][:-1] + url,
            params=params,
            directory=directory,
            filename=filename,
            headers=headers,
            zipped=file_type in ZIPPED_FILE_TYPES,
        )
        return directory / filename
