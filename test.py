import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import ecoinvent_interface as ei
from time import sleep
from tqdm import tqdm
import argparse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcoinventDatabaseCreator:
    def __init__(self, db_path: str = "ecoinvent.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Create connection to SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
    def create_tables(self):
        """Create all necessary tables in the database."""
        # Core dataset information with status tracking
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY,
                version TEXT,
                system_model TEXT,
                activity_name TEXT,
                geography_short TEXT,
                geography_long TEXT,
                geography_comment TEXT,
                reference_product TEXT,
                unit TEXT,
                sector TEXT,
                special_activity_type INTEGER,
                time_period TEXT,
                time_period_comment TEXT,
                technology_comment TEXT,
                technology_level INTEGER,
                included_activities_start TEXT,
                included_activities_end TEXT,
                basic_info_complete BOOLEAN DEFAULT FALSE,
                documentation_complete BOOLEAN DEFAULT FALSE,
                exchanges_complete BOOLEAN DEFAULT FALSE,
                impacts_complete BOOLEAN DEFAULT FALSE,
                lci_complete BOOLEAN DEFAULT FALSE,
                direct_contributions_complete BOOLEAN DEFAULT FALSE,
                consuming_activities_complete BOOLEAN DEFAULT FALSE,
                related_datasets_complete BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Track initialization status
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS initialization_status (
                id INTEGER PRIMARY KEY,
                methods_initialized BOOLEAN DEFAULT FALSE,
                categories_initialized BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Insert initialization status if not exists
        self.cursor.execute("""
            INSERT OR IGNORE INTO initialization_status (id, methods_initialized, categories_initialized)
            VALUES (1, FALSE, FALSE)
        """)
        
        # Methods table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS methods (
                id INTEGER PRIMARY KEY,
                method_name TEXT,
                UNIQUE(method_name)
            )
        """)
        
        # Impact categories within methods
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS impact_categories (
                id INTEGER PRIMARY KEY,
                method_id INTEGER,
                category_name TEXT,
                indicator_name TEXT,
                unit_name TEXT,
                FOREIGN KEY (method_id) REFERENCES methods(id),
                UNIQUE(method_id, category_name, indicator_name)
            )
        """)
        
        # Impact results
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS impact_results (
                dataset_id INTEGER,
                category_id INTEGER,
                amount REAL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id),
                FOREIGN KEY (category_id) REFERENCES impact_categories(id),
                PRIMARY KEY (dataset_id, category_id)
            )
        """)
        
        # Direct contributions
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS direct_contributions (
                dataset_id INTEGER,
                category_id INTEGER,
                contribution_type TEXT,
                amount REAL,
                unit TEXT,
                factor REAL,
                impact REAL,
                relative_contribution REAL,
                meta_name TEXT,
                meta_comp TEXT,
                meta_subcomp TEXT,
                meta_index INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id),
                FOREIGN KEY (category_id) REFERENCES impact_categories(id)
            )
        """)
        
        # Consuming activities
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS consuming_activities (
                dataset_id INTEGER,
                consuming_activity_name TEXT,
                geography TEXT,
                reference_product TEXT,
                amount REAL,
                unit_name TEXT,
                spold_id INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
        """)
        
        # Related datasets
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS related_datasets (
                dataset_id INTEGER,
                spold_id INTEGER,
                version TEXT,
                system_model TEXT,
                description TEXT,
                related_dataset_type TEXT,
                is_current BOOLEAN,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
        """)
        
        # Documentation - Reviews
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                dataset_id INTEGER,
                review_date TEXT,
                reviewer_id TEXT,
                reviewer_name TEXT,
                reviewer_email TEXT,
                reviewed_major_release INTEGER,
                reviewed_minor_release INTEGER,
                reviewed_major_revision INTEGER,
                reviewed_minor_revision INTEGER,
                comments TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
        """)
        
        # Exchanges
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER,
                exchange_name TEXT,
                exchange_group TEXT,
                comment TEXT,
                amount REAL,
                unit TEXT,
                spold_id INTEGER,
                geography TEXT,
                is_input BOOLEAN,
                uncertainty_type TEXT,
                uncertainty_pedigree_matrix TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
        """)
        
        # Exchange properties
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS exchange_properties (
                exchange_id INTEGER,
                property_name TEXT,
                unit TEXT,
                amount REAL,
                comment TEXT,
                FOREIGN KEY (exchange_id) REFERENCES exchanges(id)
            )
        """)
        
        # LCI Results
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS lci_results (
                dataset_id INTEGER,
                substance_name TEXT,
                amount REAL,
                compartment TEXT,
                sub_compartment TEXT,
                unit TEXT,
                score REAL,
                relative_contribution REAL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
        """)

        self.conn.commit()

    def is_initialized(self) -> tuple:
        """Check if methods and categories are initialized."""
        self.cursor.execute("""
            SELECT methods_initialized, categories_initialized 
            FROM initialization_status 
            WHERE id = 1
        """)
        return self.cursor.fetchone()

    def set_initialized(self, methods: bool = None, categories: bool = None):
        """Set initialization status."""
        if methods is not None:
            self.cursor.execute("""
                UPDATE initialization_status 
                SET methods_initialized = ?
                WHERE id = 1
            """, (methods,))
        if categories is not None:
            self.cursor.execute("""
                UPDATE initialization_status 
                SET categories_initialized = ?
                WHERE id = 1
            """, (categories,))
        self.conn.commit()

    def get_datasets_missing_data(self, data_type: str, limit: Optional[int] = None) -> List[int]:
        """Get list of dataset IDs missing specific data.
        
        Args:
            data_type: Type of data to check ('basic_info', 'exchanges', 'lci', etc.)
            limit: Optional limit on number of datasets to return
            
        Returns:
            List of dataset IDs missing the specified data type
        """
        # Map data types to their corresponding status columns
        status_columns = {
            'basic_info': 'basic_info_complete',
            'documentation': 'documentation_complete',
            'exchanges': 'exchanges_complete',
            'impacts': 'impacts_complete',
            'lci': 'lci_complete',
            'direct_contributions': 'direct_contributions_complete',
            'consuming_activities': 'consuming_activities_complete',
            'related_datasets': 'related_datasets_complete'
        }
        
        if data_type not in status_columns:
            raise ValueError(f"Invalid data type. Must be one of: {', '.join(status_columns.keys())}")
        
        status_column = status_columns[data_type]
        
        query = f"""
            SELECT id FROM datasets 
            WHERE {status_column} = FALSE OR {status_column} IS NULL
            ORDER BY id
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        self.cursor.execute(query)
        return [row[0] for row in self.cursor.fetchall()]

    def get_dataset_status(self, dataset_id: int) -> Dict[str, bool]:
        """Get completion status for all data types for a specific dataset.
        
        Args:
            dataset_id: ID of the dataset to check
            
        Returns:
            Dictionary mapping data types to their completion status
        """
        self.cursor.execute("""
            SELECT basic_info_complete,
                documentation_complete,
                exchanges_complete,
                impacts_complete,
                lci_complete,
                direct_contributions_complete,
                consuming_activities_complete,
                related_datasets_complete
            FROM datasets
            WHERE id = ?
        """, (dataset_id,))
        
        row = self.cursor.fetchone()
        if not row:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        return {
            'basic_info': bool(row[0]),
            'documentation': bool(row[1]),
            'exchanges': bool(row[2]),
            'impacts': bool(row[3]),
            'lci': bool(row[4]),
            'direct_contributions': bool(row[5]),
            'consuming_activities': bool(row[6]),
            'related_datasets': bool(row[7])
        }

    def mark_data_complete(self, dataset_id: int, data_type: str):
        """Mark a specific data type as complete for a dataset.
        
        Args:
            dataset_id: ID of the dataset to update
            data_type: Type of data to mark as complete
        """
        status_columns = {
            'basic_info': 'basic_info_complete',
            'documentation': 'documentation_complete',
            'exchanges': 'exchanges_complete',
            'impacts': 'impacts_complete',
            'lci': 'lci_complete',
            'direct_contributions': 'direct_contributions_complete',
            'consuming_activities': 'consuming_activities_complete',
            'related_datasets': 'related_datasets_complete'
        }
        
        if data_type not in status_columns:
            raise ValueError(f"Invalid data type. Must be one of: {', '.join(status_columns.keys())}")
        
        status_column = status_columns[data_type]
        
        self.cursor.execute(f"""
            UPDATE datasets 
            SET {status_column} = TRUE 
            WHERE id = ?
        """, (dataset_id,))
        self.conn.commit()

    def process_dataset_lci(self, ep, dataset_id: int):
        """Process LCI results for a dataset."""
        try:
            # Get LCI results
            lci_results = self.rate_limited_api_call(ep.get_lci)
            if lci_results:
                self.insert_lci_results(dataset_id, lci_results)
                self.mark_data_complete(dataset_id, 'lci')
            return True
        except Exception as e:
            logger.error(f"Error processing LCI results for dataset {dataset_id}: {str(e)}")
            return False

    def process_consuming_activities(self, ep, dataset_id: int):
        """Process consuming activities for a dataset."""
        try:
            # Get consuming activities
            activities = self.rate_limited_api_call(ep.get_consuming_activities)
            if activities:
                self.insert_consuming_activities(dataset_id, activities)
                self.mark_data_complete(dataset_id, 'consuming_activities')
            return True
        except Exception as e:
            logger.error(f"Error processing consuming activities for dataset {dataset_id}: {str(e)}")
            return False

    def process_related_datasets(self, ep, dataset_id: int):
        """Process related datasets for a dataset."""
        try:
            # Get related datasets
            related = self.rate_limited_api_call(ep.get_related_datasets)
            if related:
                self.insert_related_datasets(dataset_id, related)
                self.mark_data_complete(dataset_id, 'related_datasets')
            return True
        except Exception as e:
            logger.error(f"Error processing related datasets for dataset {dataset_id}: {str(e)}")
            return False



    def insert_dataset_basic_info(self, dataset_id: int, basic_info: Dict, documentation: Dict):
        """Insert basic dataset and documentation information."""
        activity_desc = documentation.get('activity_description', {})
        self.cursor.execute("""
            INSERT OR REPLACE INTO datasets 
            (id, version, system_model, activity_name, geography_short, 
             geography_long, geography_comment, reference_product, unit, sector,
             special_activity_type, time_period, time_period_comment,
             technology_comment, technology_level, included_activities_start,
             included_activities_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_id,
            basic_info['version'],
            basic_info['system_model'],
            basic_info['activity_name'],
            basic_info['geography']['short_name'],
            basic_info['geography'].get('long_name'),
            activity_desc.get('geography', {}).get('comment'),
            basic_info['reference_product'],
            basic_info.get('unit'),
            basic_info.get('sector'),
            activity_desc.get('special_activity_type'),
            activity_desc.get('time_period'),
            activity_desc.get('time_period_comment'),
            activity_desc.get('technology', {}).get('comment'),
            activity_desc.get('technology', {}).get('technology_level'),
            activity_desc.get('included_activities_start'),
            activity_desc.get('included_activities_end')
        ))
        self.conn.commit()

        # Insert reviews if present
        reviews = documentation.get('modelling_and_validation', {}).get('reviews', [])
        for review in reviews:
            self.cursor.execute("""
                INSERT INTO reviews
                (dataset_id, review_date, reviewer_id, reviewer_name, reviewer_email,
                 reviewed_major_release, reviewed_minor_release,
                 reviewed_major_revision, reviewed_minor_revision, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                review.get('review_date'),
                review.get('reviewer_id'),
                review.get('reviewer_name'),
                review.get('reviewer_email'),
                review.get('reviewed_major_release'),
                review.get('reviewed_minor_release'),
                review.get('reviewed_major_revision'),
                review.get('reviewed_minor_revision'),
                review.get('comments')
            ))
        self.conn.commit()

    def insert_method(self, method_info: Dict) -> int:
        """Insert method information and return its ID."""
        self.cursor.execute("""
            INSERT OR IGNORE INTO methods (id, method_name)
            VALUES (?, ?)
        """, (method_info['method_id'], method_info['method_name']))
        self.conn.commit()
        return method_info['method_id']

    def insert_impact_category(self, method_id: int, impact_info: Dict) -> int:
        """Insert impact category information and return its ID."""
        self.cursor.execute("""
            INSERT OR IGNORE INTO impact_categories 
            (id, method_id, category_name, indicator_name, unit_name)
            VALUES (?, ?, ?, ?, ?)
        """, (
            impact_info['index'],
            method_id,
            impact_info['category_name'],
            impact_info['indicator_name'],
            impact_info['unit_name']
        ))
        self.conn.commit()
        return impact_info['index']

    def insert_direct_contributions(self, dataset_id: int, category_id: int, contributions: List[Dict]):
        """Insert direct contribution results."""
        for contrib in contributions:
            self.cursor.execute("""
                INSERT INTO direct_contributions 
                (dataset_id, category_id, contribution_type, amount, unit,
                 factor, impact, relative_contribution, meta_name, meta_comp,
                 meta_subcomp, meta_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                category_id,
                contrib['type'],
                contrib['amount'],
                contrib['unit'],
                contrib['factor'],
                contrib['impact'],
                contrib['relative_contribution'],
                contrib['meta']['name'],
                contrib['meta']['comp'],
                contrib['meta']['subcomp'],
                contrib['meta'].get('index')
            ))
        self.conn.commit()

    def insert_exchanges(self, dataset_id: int, exchanges: Dict):
        """Insert exchanges information."""
        for exchange_type in ['intermediateExchange']:
            for exchange in exchanges.get(exchange_type, []):
                self.cursor.execute("""
                    INSERT INTO exchanges
                    (dataset_id, exchange_name, exchange_group, comment,
                     amount, unit, spold_id, geography, is_input,
                     uncertainty_type, uncertainty_pedigree_matrix)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dataset_id,
                    exchange['name'],
                    exchange['group'],
                    exchange.get('comment'),
                    exchange['amount'],
                    exchange.get('unit'),
                    exchange.get('link', {}).get('url'),
                    exchange.get('link', {}).get('geography'),
                    exchange_type == 'intermediateExchange',
                    exchange.get('uncertainty', {}).get('type'),
                    exchange.get('uncertainty', {}).get('pedigreeMatrix')
                ))
                
                exchange_id = self.cursor.lastrowid
                
                # Insert properties
                for prop in exchange.get('property', []):
                    self.cursor.execute("""
                        INSERT INTO exchange_properties
                        (exchange_id, property_name, unit, amount, comment)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        exchange_id,
                        prop['name'],
                        prop.get('unit'),
                        prop.get('amount'),
                        prop.get('comment')
                    ))
        
        self.conn.commit()

    def insert_lci_results(self, dataset_id: int, results: Dict):
        """Insert LCI results."""
        for result in results.get('lci_results', []):
            self.cursor.execute("""
                INSERT INTO lci_results
                (dataset_id, substance_name, amount, compartment,
                 sub_compartment, unit, score, relative_contribution)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                result['name'],
                result['amount'],
                result.get('compartment'),
                result.get('sub_compartment'),
                result.get('unit'),
                result.get('score'),
                result.get('relative_contribution')
            ))
        self.conn.commit()

    def insert_impact_result(self, dataset_id: int, category_id: int, amount: float):
        """Insert impact assessment result."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO impact_results (dataset_id, category_id, amount)
            VALUES (?, ?, ?)
        """, (dataset_id, category_id, amount))
        self.conn.commit()

    def insert_consuming_activities(self, dataset_id: int, activities: List[Dict]):
        """Insert consuming activities for a dataset."""
        for activity in activities:
            self.cursor.execute("""
                INSERT INTO consuming_activities 
                (dataset_id, consuming_activity_name, geography, reference_product, 
                 amount, unit_name, spold_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                activity['activity_name'],
                activity['geography'],
                activity['reference_product'],
                activity['amount'],
                activity['unit_name'],
                activity['spold_id']
            ))
        self.conn.commit()

    def rate_limited_api_call(self, call_function, *args, max_retries=3, base_delay=2.0):
        """Execute an API call with rate limit handling and retries.
        
        Args:
            call_function: Function to call
            *args: Arguments to pass to the function
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            
        Returns:
            Result from the API call
        """
        for attempt in range(max_retries):
            try:
                return call_function(*args)
            except Exception as e:
                if "Rate limit exceeded" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit, waiting {delay} seconds before retry...")
                    sleep(delay)
                    continue
                raise

    def process_dataset_basics(self, ep, dataset_id: int):
        """Process only basic dataset information with rate limiting."""
        try:
            # Get basic info and documentation
            basic_info = self.rate_limited_api_call(ep.get_basic_info)
            if not basic_info:
                raise ValueError(f"Could not get basic info for dataset {dataset_id}")
            sleep(1.5)  # Add delay between API calls
            
            documentation = self.rate_limited_api_call(ep.get_documentation)
            if not documentation:
                raise ValueError(f"Could not get documentation for dataset {dataset_id}")
            
            self.insert_dataset_basic_info(dataset_id, basic_info, documentation)
            self.mark_data_complete(dataset_id, 'basic_info')
            self.mark_data_complete(dataset_id, 'documentation')
            sleep(1.5)
            
            # Get methods and process impacts
            methods = self.rate_limited_api_call(ep.get_methods)
            if methods:
                for method in methods:
                    impacts = self.rate_limited_api_call(ep.get_impacts, str(method['method_id']))
                    if impacts:
                        for impact in impacts:
                            category_id = impact['index']
                            self.insert_impact_result(dataset_id, category_id, impact['amount'])
                    sleep(1.5)  # Add delay between methods
                self.mark_data_complete(dataset_id, 'impacts')
                
            return True
                
        except Exception as e:
            logger.error(f"Error processing core data for dataset {dataset_id}: {str(e)}")
            return False

    def process_dataset_exchanges(self, ep, dataset_id: int):
        """Process only exchanges for a dataset."""
        try:
            # Get exchanges
            exchanges = self.rate_limited_api_call(ep.get_exchanges)
            if exchanges and isinstance(exchanges, dict):
                self.insert_exchanges(dataset_id, exchanges)
                self.mark_data_complete(dataset_id, 'exchanges')
            sleep(1.5)
            return True
        except Exception as e:
            logger.error(f"Error processing exchanges for dataset {dataset_id}: {str(e)}")
            return False

    def get_datasets_without_exchanges(self) -> List[int]:
        """Get list of dataset IDs that don't have exchanges."""
        self.cursor.execute("""
            SELECT DISTINCT d.id 
            FROM datasets d
            LEFT JOIN exchanges e ON d.id = e.dataset_id
            WHERE e.id IS NULL
        """)
        return [row[0] for row in self.cursor.fetchall()]

    def add_exchanges_to_existing_datasets(self, ep: ei.EcoinventProcess, limit: Optional[int] = None):
        """Add exchanges to datasets that don't have them yet."""
        datasets = self.get_datasets_without_exchanges()
        if limit:
            datasets = datasets[:limit]
            
        logger.info(f"Found {len(datasets)} datasets without exchanges")
        
        for dataset_id in tqdm(datasets, desc="Adding exchanges"):
            try:
                # Select the process
                ep.select_process(dataset_id=str(dataset_id))
                
                # Process exchanges
                success = self.process_dataset_exchanges(ep, dataset_id)
                if success:
                    logger.info(f"Successfully added exchanges for dataset {dataset_id}")
                else:
                    logger.warning(f"Failed to add exchanges for dataset {dataset_id}")
                
                sleep(1.5)  # Add delay between datasets
                
            except Exception as e:
                logger.error(f"Error adding exchanges for dataset {dataset_id}: {str(e)}")
                continue
        """Insert related datasets information."""
        for dataset in related:
            self.cursor.execute("""
                INSERT INTO related_datasets 
                (dataset_id, spold_id, version, system_model, description, 
                 related_dataset_type, is_current)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                dataset['spold_id'],
                dataset['version'],
                dataset['system_model'],
                dataset['description'],
                dataset['related_dataset_type'],
                dataset['is_current']
            ))
        self.conn.commit()

    def initialize_methods_and_categories(self, ep: ei.EcoinventProcess):
        """Initialize methods and impact categories once."""
        methods_init, categories_init = self.is_initialized()
        
        if not methods_init or not categories_init:
            # Select any dataset to get methods
            dataset_id = "1"  # Use first dataset
            ep.select_process(dataset_id=dataset_id)
            
            if not methods_init:
                logger.info("Initializing methods...")
                methods = ep.get_methods()
                for method in methods:
                    self.insert_method(method)
                self.set_initialized(methods=True)
                
            if not categories_init:
                logger.info("Initializing impact categories...")
                methods = ep.get_methods()
                for method in methods:
                    impacts = ep.get_impacts(str(method['method_id']))
                    for impact in impacts:
                        self.insert_impact_category(method['method_id'], impact)
                self.set_initialized(categories=True)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None


def main():
    parser = argparse.ArgumentParser(description='Create ecoinvent database')
    parser.add_argument('--limit', type=int, help='Limit number of datasets to process')
    parser.add_argument('--start-from', type=int, help='Start processing from this dataset ID')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for rate-limited calls')
    parser.add_argument('--base-delay', type=float, default=1.5, help='Base delay between API calls')
    parser.add_argument('--add-exchanges', action='store_true', help='Add exchanges to existing datasets')
    args = parser.parse_args()

    # Initialize ecoinvent interface
    try:
        my_settings = ei.Settings()
        ep = ei.EcoinventProcess(my_settings)
        ep.set_release(version="3.10.1", system_model="EN15804")
    except Exception as e:
        logger.error(f"Failed to initialize ecoinvent interface: {str(e)}")
        return

    # Create database
    db_creator = EcoinventDatabaseCreator("ecoinvent_3.10.1_EN15804.db")
    db_creator.connect()
    db_creator.create_tables()
    
    try:
        if args.add_exchanges:
            # Only add exchanges to existing datasets
            db_creator.add_exchanges_to_existing_datasets(ep, args.limit)
            return

        # Initialize methods and categories once
        db_creator.initialize_methods_and_categories(ep)
        
        # Read mapping file
        with open('./ecoinvent_interface/data/mappings/3.10_EN15804.json', 'r') as f:
            datasets = json.load(f)
        
        # Get list of incomplete datasets
        incomplete_datasets = set(db_creator.get_incomplete_datasets())
        
        # If no incomplete datasets, start fresh
        if not incomplete_datasets:
            incomplete_datasets = {dataset['index'] for dataset in datasets}
        
        # Filter datasets to process
        datasets_to_process = [
            dataset for dataset in datasets 
            if dataset['index'] in incomplete_datasets
        ]
        
        # Apply start-from filter if specified
        if args.start_from:
            datasets_to_process = [
                dataset for dataset in datasets_to_process 
                if dataset['index'] >= args.start_from
            ]
        
        # Apply limit if specified
        if args.limit:
            datasets_to_process = datasets_to_process[:args.limit]
        
        # Process each dataset
        for dataset in tqdm(datasets_to_process, desc="Processing datasets"):
            try:
                dataset_id = dataset['index']
                logger.info(f"Processing dataset {dataset_id}")
                
                # Select the process
                ep.select_process(dataset_id=str(dataset_id))
                
                # Process the dataset basic info
                db_creator.process_dataset_basics(ep, dataset_id)
                
                # Mark dataset as complete
                db_creator.mark_dataset_complete(dataset_id)
                
                # Add a small delay between datasets
                sleep(args.base_delay)
                
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_id}: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    
    finally:
        db_creator.close()
        logger.info("Database creation process finished!")

def main():
    parser = argparse.ArgumentParser(description='Create ecoinvent database')
    parser.add_argument('--limit', type=int, help='Limit number of datasets to process')
    parser.add_argument('--start-from', type=int, help='Start processing from this dataset ID')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for rate-limited calls')
    parser.add_argument('--base-delay', type=float, default=1.5, help='Base delay between API calls')
    
    # Add data type flags
    parser.add_argument('--add-basics', action='store_true', help='Add basic info and impacts')
    parser.add_argument('--add-exchanges', action='store_true', help='Add exchanges to datasets')
    parser.add_argument('--add-lci', action='store_true', help='Add LCI results to datasets')
    parser.add_argument('--add-consuming', action='store_true', help='Add consuming activities to datasets')
    parser.add_argument('--add-related', action='store_true', help='Add related datasets')
    parser.add_argument('--show-status', type=int, help='Show status for a specific dataset ID')
    args = parser.parse_args()

    # Initialize ecoinvent interface
    try:
        my_settings = ei.Settings()
        ep = ei.EcoinventProcess(my_settings)
        ep.set_release(version="3.10.1", system_model="EN15804")
    except Exception as e:
        logger.error(f"Failed to initialize ecoinvent interface: {str(e)}")
        return

    # Create database
    db_creator = EcoinventDatabaseCreator("ecoinvent_3.10.1_EN15804.db")
    db_creator.connect()
    db_creator.create_tables()
    
    try:
        # Show status for a specific dataset if requested
        if args.show_status:
            status = db_creator.get_dataset_status(args.show_status)
            print(f"\nStatus for dataset {args.show_status}:")
            for field, value in status.items():
                print(f"  {field}: {'Complete' if value else 'Incomplete'}")
            return

        # Initialize methods and categories once
        db_creator.initialize_methods_and_categories(ep)
        
        if args.add_basics:
            # Process basic info and impacts
            with open('./ecoinvent_interface/data/mappings/3.10_EN15804.json', 'r') as f:
                datasets = json.load(f)
            
            # Get datasets missing basic info
            incomplete = db_creator.get_datasets_missing_data('basic_info', args.limit)
            datasets_to_process = [d for d in datasets if d['index'] in incomplete]
            
            for dataset in tqdm(datasets_to_process, desc="Processing basics"):
                try:
                    dataset_id = dataset['index']
                    logger.info(f"Processing basics for dataset {dataset_id}")
                    ep.select_process(dataset_id=str(dataset_id))
                    db_creator.process_dataset_basics(ep, dataset_id)
                    sleep(args.base_delay)
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset_id}: {str(e)}")
                    continue
                    
        elif args.add_exchanges:
            # Add exchanges to existing datasets
            datasets = db_creator.get_datasets_missing_data('exchanges', args.limit)
            for dataset_id in tqdm(datasets, desc="Adding exchanges"):
                try:
                    ep.select_process(dataset_id=str(dataset_id))
                    db_creator.process_dataset_exchanges(ep, dataset_id)
                    sleep(args.base_delay)
                except Exception as e:
                    logger.error(f"Error adding exchanges for dataset {dataset_id}: {str(e)}")
                    continue
                    
        elif args.add_lci:
            # Add LCI results to existing datasets
            datasets = db_creator.get_datasets_missing_data('lci', args.limit)
            for dataset_id in tqdm(datasets, desc="Adding LCI results"):
                try:
                    ep.select_process(dataset_id=str(dataset_id))
                    db_creator.process_dataset_lci(ep, dataset_id)
                    sleep(args.base_delay)
                except Exception as e:
                    logger.error(f"Error adding LCI results for dataset {dataset_id}: {str(e)}")
                    continue
                    
        elif args.add_consuming:
            # Add consuming activities to existing datasets
            datasets = db_creator.get_datasets_missing_data('consuming_activities', args.limit)
            for dataset_id in tqdm(datasets, desc="Adding consuming activities"):
                try:
                    ep.select_process(dataset_id=str(dataset_id))
                    db_creator.process_consuming_activities(ep, dataset_id)
                    sleep(args.base_delay)
                except Exception as e:
                    logger.error(f"Error adding consuming activities for dataset {dataset_id}: {str(e)}")
                    continue
                    
        elif args.add_related:
            # Add related datasets to existing datasets
            datasets = db_creator.get_datasets_missing_data('related_datasets', args.limit)
            for dataset_id in tqdm(datasets, desc="Adding related datasets"):
                try:
                    ep.select_process(dataset_id=str(dataset_id))
                    db_creator.process_related_datasets(ep, dataset_id)
                    sleep(args.base_delay)
                except Exception as e:
                    logger.error(f"Error adding related datasets for dataset {dataset_id}: {str(e)}")
                    continue
                    
        else:
            logger.info("Please specify what data to add using one of the --add-* options")
            logger.info("Available options:")
            logger.info("  --add-basics: Add basic info and impacts")
            logger.info("  --add-exchanges: Add exchanges")
            logger.info("  --add-lci: Add LCI results")
            logger.info("  --add-consuming: Add consuming activities")
            logger.info("  --add-related: Add related datasets")
            logger.info("  --show-status DATASET_ID: Show status for a specific dataset")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    
    finally:
        db_creator.close()
        logger.info("Database creation process finished!")

if __name__ == "__main__":
    main()