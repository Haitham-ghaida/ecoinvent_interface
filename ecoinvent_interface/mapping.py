import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from time import sleep
from typing import List, Optional

import pyecospold
from tqdm import tqdm

from . import CachedStorage, EcoinventProcess, Settings
from .core import SYSTEM_MODELS
from .release import ReleaseType

DATA_DIR = Path(__file__).parent.resolve() / "data"


def get_rp_text(exchanges: list) -> str:
    rp_exchanges = [
        exc for exc in exchanges if exc.groupStr == "ReferenceProduct" and exc.amount
    ]
    if len(rp_exchanges) != 1:
        raise ValueError("Can't find single reference product")
    return rp_exchanges[0].names[0]


class ProcessMapping:
    def __init__(
        self, settings: Settings, storage: Optional[CachedStorage] = None
    ) -> None:
        self.settings = settings
        self.storage = storage or CachedStorage()

    def create_remote_mapping(
        self, version: str, system_model: str, max_id: int
    ) -> list:
        remote_data = []

        process = EcoinventProcess(self.settings)
        process.set_release(version, system_model)

        for index in tqdm(range(1, max_id + 1)):
            process.dataset_id = index
            remote_data.append(process.get_basic_info())
            sleep(0.1)

        return remote_data

    def create_local_mapping(self, version: str, system_model: str) -> None:
        key = ReleaseType.ecospold.filename(
            version=version,
            system_model_abbr=SYSTEM_MODELS.get(system_model, system_model),
        )
        if key not in self.storage.catalogue:
            ERROR = f"{key} not in current catalogue. Download the release and retry."
            raise ValueError(ERROR)

        dir_path = Path(self.storage.catalogue[key]["path"]) / "datasets"
        local_data = []
        file_paths = [
            fp
            for fp in dir_path.iterdir()
            if fp.is_file() and fp.suffix.lower() == ".spold"
        ]

        for file_path in tqdm(file_paths):
            ecospold = pyecospold.parse_file_v2(file_path)
            local_data.append(
                {
                    "filename": file_path.name,
                    "activity_name": ecospold.activityDataset.activityDescription.activity[  # NOQA E501
                        0
                    ].activityNames[
                        0
                    ],
                    "reference_product": get_rp_text(
                        ecospold.activityDataset.flowData.intermediateExchanges
                    ),
                    "geography": ecospold.activityDataset.activityDescription.geography[
                        0
                    ].shortNames[0],
                }
            )

        return local_data

    def add_mapping(self, data: List[dict], version: str, system_model: str) -> Path:
        zf = zipfile.ZipFile(DATA_DIR / "mappings.zip")

        with tempfile.TemporaryDirectory() as td:
            zf.extractall(path=td)
            metadata = json.load(open(Path(td) / "catalogue.json"))
            assert not any(
                obj["version"] == version and obj["system_model"] == system_model
                for obj in metadata
            )
            filename = f"{version}_{system_model}.json"
            metadata.append(
                {
                    "filename": filename,
                    "version": version,
                    "system_model": system_model,
                }
            )

            with open(Path(td) / "catalogue.json", "w") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            with open(Path(td) / filename, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            shutil.copyfile(DATA_DIR / "mappings.zip", DATA_DIR / "mappings.backup.zip")
            shutil.make_archive(DATA_DIR / "mappings", "zip", td)

            with zipfile.ZipFile(
                DATA_DIR / "mappings.zip", "w", zipfile.ZIP_LZMA
            ) as archive:
                for file_path in Path(td).iterdir():
                    if not file_path.is_file():
                        continue
                    archive.write(file_path, arcname=file_path.name)

        return DATA_DIR / "mappings.zip"


from typing import Dict, List, Set
import logging
from collections import defaultdict

class ImpactIndexMapper:
    """Maps and explores ecoinvent impact assessment method indices"""
    
    def __init__(self, process):
        """Initialize with an EcoinventProcess instance"""
        self.process = process
        self.found_indices = set()
        self.method_groups = defaultdict(set)  # Method name -> set of indices
        self.index_metadata = {}  # Index -> full metadata
        
    def explore_index(self, start_index: int) -> List[dict]:
        """Explore a specific index and its related indices
        
        Args:
            start_index: Index number to explore
            
        Returns:
            List of impact results found
        """
        try:
            results = self.process.get_impacts(str(start_index))
            
            # Make sure results is a list
            if not isinstance(results, list):
                logging.warning(f"Unexpected response type for index {start_index}: {type(results)}")
                return []
                
            # Store results data
            for result in results:
                if not isinstance(result, dict):
                    continue
                    
                index = result.get('index')
                if index is None:
                    continue
                    
                method_name = result.get('method_name')
                if method_name:
                    self.method_groups[method_name].add(index)
                    
                self.found_indices.add(index)
                self.index_metadata[index] = {
                    'method_name': result.get('method_name'),
                    'category_name': result.get('category_name'),
                    'indicator_name': result.get('indicator_name'),
                    'unit_name': result.get('unit_name')
                }
            
            return results
            
        except Exception as e:
            logging.warning(f"Error exploring index {start_index}: {str(e)}")
            return []
            
    def scan_range(self, start: int, end: int, step: int = 1) -> Dict[str, Set[int]]:
        """Scan a range of indices to build method mapping
        
        Args:
            start: Starting index
            end: Ending index
            step: Step size for scanning
            
        Returns:
            Dict mapping method names to sets of indices
        """
        for i in range(start, end + 1, step):
            self.explore_index(i)
            
        return dict(self.method_groups)  # Convert defaultdict to regular dict
        
    def get_index_info(self, index: int) -> dict:
        """Get stored metadata for a specific index"""
        return self.index_metadata.get(index)
        
    def list_methods(self) -> List[str]:
        """Get list of all found method names"""
        return list(self.method_groups.keys())
        
    def get_indices_for_method(self, method_name: str) -> Set[int]:
        """Get all indices associated with a method name"""
        return self.method_groups.get(method_name, set())
        
    def get_method_structure(self) -> Dict[str, Dict[str, List[dict]]]:
        """Get hierarchical structure of methods and categories
        
        Returns:
            Dict with structure:
            {
                method_name: {
                    category_name: [
                        {
                            'index': index,
                            'indicator': indicator_name,
                            'unit': unit_name
                        },
                        ...
                    ],
                    ...
                },
                ...
            }
        """
        structure = defaultdict(lambda: defaultdict(list))
        
        for index, meta in self.index_metadata.items():
            method = meta['method_name']
            category = meta['category_name']
            structure[method][category].append({
                'index': index,
                'indicator': meta['indicator_name'],
                'unit': meta['unit_name']
            })
            
        return dict(structure)  # Convert to regular dict