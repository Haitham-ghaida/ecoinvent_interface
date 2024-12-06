import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
from time import sleep
from datetime import datetime

import ecoinvent_interface as ei
from tqdm import tqdm
import argparse
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class Dataset(Base):
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    version = Column(String)
    system_model = Column(String)
    activity_name = Column(String)
    geography_short = Column(String)
    geography_long = Column(String)
    geography_comment = Column(String)
    reference_product = Column(String)
    unit = Column(String)
    sector = Column(String)
    special_activity_type = Column(Integer)
    time_period = Column(String)
    time_period_comment = Column(String)
    technology_comment = Column(String)
    technology_level = Column(Integer)
    included_activities_start = Column(String)
    included_activities_end = Column(String)
    
    # Status flags
    basic_info_complete = Column(Boolean, default=False)
    documentation_complete = Column(Boolean, default=False)
    exchanges_complete = Column(Boolean, default=False)
    impacts_complete = Column(Boolean, default=False)
    lci_complete = Column(Boolean, default=False)
    direct_contributions_complete = Column(Boolean, default=False)
    consuming_activities_complete = Column(Boolean, default=False)
    related_datasets_complete = Column(Boolean, default=False)
    
    # Relationships
    reviews = relationship("Review", back_populates="dataset")
    exchanges = relationship("Exchange", back_populates="dataset")
    impact_results = relationship("ImpactResult", back_populates="dataset")
    lci_results = relationship("LCIResult", back_populates="dataset")
    consuming_activities = relationship("ConsumingActivity", back_populates="dataset")
    related_datasets = relationship("RelatedDataset", back_populates="dataset")

class Review(Base):
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    review_date = Column(String)
    reviewer_id = Column(String)
    reviewer_name = Column(String)
    reviewer_email = Column(String)
    reviewed_major_release = Column(Integer)
    reviewed_minor_release = Column(Integer)
    reviewed_major_revision = Column(Integer)
    reviewed_minor_revision = Column(Integer)
    comments = Column(String)
    
    dataset = relationship("Dataset", back_populates="reviews")

class Method(Base):
    __tablename__ = 'methods'
    
    id = Column(Integer, primary_key=True)
    method_name = Column(String, unique=True)
    impact_categories = relationship("ImpactCategory", back_populates="method")

class ImpactCategory(Base):
    __tablename__ = 'impact_categories'
    
    id = Column(Integer, primary_key=True)
    method_id = Column(Integer, ForeignKey('methods.id'))
    category_name = Column(String)
    indicator_name = Column(String)
    unit_name = Column(String)
    
    method = relationship("Method", back_populates="impact_categories")
    impact_results = relationship("ImpactResult", back_populates="category")

class ImpactResult(Base):
    __tablename__ = 'impact_results'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    category_id = Column(Integer, ForeignKey('impact_categories.id'))
    amount = Column(Float)
    
    dataset = relationship("Dataset", back_populates="impact_results")
    category = relationship("ImpactCategory", back_populates="impact_results")

class Exchange(Base):
    __tablename__ = 'exchanges'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    exchange_name = Column(String)
    exchange_group = Column(String)
    comment = Column(String)
    amount = Column(Float)
    unit = Column(String)
    spold_id = Column(Integer)
    geography = Column(String)
    is_input = Column(Boolean)
    uncertainty_type = Column(String)
    uncertainty_pedigree_matrix = Column(String)
    
    dataset = relationship("Dataset", back_populates="exchanges")
    properties = relationship("ExchangeProperty", back_populates="exchange")

class ExchangeProperty(Base):
    __tablename__ = 'exchange_properties'
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'))
    property_name = Column(String)
    unit = Column(String)
    amount = Column(Float)
    comment = Column(String)
    
    exchange = relationship("Exchange", back_populates="properties")

class LCIResult(Base):
    __tablename__ = 'lci_results'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    substance_name = Column(String)
    amount = Column(Float)
    compartment = Column(String)
    sub_compartment = Column(String)
    unit = Column(String)
    score = Column(Float)
    relative_contribution = Column(Float)
    
    dataset = relationship("Dataset", back_populates="lci_results")

class ConsumingActivity(Base):
    __tablename__ = 'consuming_activities'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    consuming_activity_name = Column(String)
    geography = Column(String)
    reference_product = Column(String)
    amount = Column(Float)
    unit_name = Column(String)
    spold_id = Column(Integer)
    
    dataset = relationship("Dataset", back_populates="consuming_activities")

class RelatedDataset(Base):
    __tablename__ = 'related_datasets'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    spold_id = Column(Integer)
    version = Column(String)
    system_model = Column(String)
    description = Column(String)
    related_dataset_type = Column(String)
    is_current = Column(Boolean)
    
    dataset = relationship("Dataset", back_populates="related_datasets")

class InitializationStatus(Base):
    __tablename__ = 'initialization_status'
    
    id = Column(Integer, primary_key=True)
    methods_initialized = Column(Boolean, default=False)
    categories_initialized = Column(Boolean, default=False)

class EcoinventDatabaseCreator:
    def __init__(self, db_path: str = "ecoinvent.db"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize status if not exists
        session = self.Session()
        if not session.query(InitializationStatus).first():
            session.add(InitializationStatus(id=1))
            session.commit()
        session.close()

    def get_datasets_missing_data(self, data_type: str, limit: Optional[int] = None) -> List[int]:
        """Get dataset IDs missing specific data type."""
        status_columns = {
            'basic_info': Dataset.basic_info_complete,
            'documentation': Dataset.documentation_complete,
            'exchanges': Dataset.exchanges_complete,
            'impacts': Dataset.impacts_complete,
            'lci': Dataset.lci_complete,
            'direct_contributions': Dataset.direct_contributions_complete,
            'consuming_activities': Dataset.consuming_activities_complete,
            'related_datasets': Dataset.related_datasets_complete
        }
        
        if data_type not in status_columns:
            raise ValueError(f"Invalid data type. Must be one of: {', '.join(status_columns.keys())}")
        
        session = self.Session()
        query = session.query(Dataset.id).filter(
            (status_columns[data_type] == False) | 
            (status_columns[data_type].is_(None))
        ).order_by(Dataset.id)
        
        if limit:
            query = query.limit(limit)
            
        result = [row[0] for row in query.all()]
        session.close()
        return result

    def ensure_dataset_exists(self, dataset_id: int, version: str, system_model: str):
        """Ensure dataset exists in database with basic info."""
        session = self.Session()
        dataset = session.query(Dataset).get(dataset_id)
        if not dataset:
            dataset = Dataset(
                id=dataset_id,
                version=version,
                system_model=system_model
            )
            session.add(dataset)
            session.commit()
        session.close()
        return dataset

    def process_dataset_basics(self, ep: ei.EcoinventProcess, dataset_id: int):
        """Process basic dataset information."""
        session = self.Session()
        try:
            # Get basic info and documentation
            basic_info = self.rate_limited_api_call(ep.get_basic_info)
            if not basic_info:
                raise ValueError(f"Could not get basic info for dataset {dataset_id}")
            sleep(1.5)
            
            documentation = self.rate_limited_api_call(ep.get_documentation)
            if not documentation:
                raise ValueError(f"Could not get documentation for dataset {dataset_id}")
            
            # Update or create dataset
            dataset = session.query(Dataset).get(dataset_id)
            if not dataset:
                dataset = Dataset(id=dataset_id)
                session.add(dataset)
            
            # Update dataset attributes
            activity_desc = documentation.get('activity_description', {})
            dataset.version = basic_info['version']
            dataset.system_model = basic_info['system_model']
            dataset.activity_name = basic_info['activity_name']
            dataset.geography_short = basic_info['geography']['short_name']
            dataset.geography_long = basic_info['geography'].get('long_name')
            dataset.geography_comment = activity_desc.get('geography', {}).get('comment')
            dataset.reference_product = basic_info['reference_product']
            dataset.unit = basic_info.get('unit')
            dataset.sector = basic_info.get('sector')
            
            # Mark as complete
            dataset.basic_info_complete = True
            dataset.documentation_complete = True
            
            session.commit()
            sleep(1.5)
            
            # Process methods and impacts
            methods = self.rate_limited_api_call(ep.get_methods)
            if methods:
                for method in methods:
                    impacts = self.rate_limited_api_call(ep.get_impacts, str(method['method_id']))
                    if impacts:
                        for impact in impacts:
                            result = ImpactResult(
                                dataset_id=dataset_id,
                                category_id=impact['index'],
                                amount=impact['amount']
                            )
                            session.add(result)
                    sleep(1.5)
                
                dataset.impacts_complete = True
                session.commit()
            
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error processing basics for dataset {dataset_id}: {str(e)}")
            return False
        finally:
            session.close()

    def rate_limited_api_call(self, call_function, *args, max_retries=3, base_delay=0.5):
        """Make rate-limited API call with retries."""
        for attempt in range(max_retries):
            try:
                return call_function(*args)
            except Exception as e:
                if "Rate limit exceeded" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, waiting {delay} seconds...")
                    sleep(delay)
                    continue
                raise

    def initialize_methods_and_categories(self, ep: ei.EcoinventProcess):
        """Initialize methods and categories."""
        session = self.Session()
        status = session.query(InitializationStatus).first()
        
        if not status.methods_initialized or not status.categories_initialized:
            ep.select_process(dataset_id="1")
            
            if not status.methods_initialized:
                logger.info("Initializing methods...")
                methods = ep.get_methods()
                for method_info in methods:
                    method = Method(
                        id=method_info['method_id'],
                        method_name=method_info['method_name']
                    )
                    session.add(method)
                status.methods_initialized = True
                session.commit()
                
            if not status.categories_initialized:
                logger.info("Initializing impact categories...")
                methods = ep.get_methods()
                for method in methods:
                    impacts = ep.get_impacts(str(method['method_id']))
                    for impact in impacts:
                        category = ImpactCategory(
                            id=impact['index'],
                            method_id=method['method_id'],
                            category_name=impact['category_name'],
                            indicator_name=impact['indicator_name'],
                            unit_name=impact['unit_name']
                        )
                        session.add(category)
                    sleep(1.5)
                status.categories_initialized = True
                session.commit()
        
        session.close()
    def process_dataset_exchanges(self, ep: ei.EcoinventProcess, dataset_id: int):
        """Process exchanges for a dataset with bulk inserts."""
        session = self.Session()
        try:
            exchanges_data = self.rate_limited_api_call(ep.get_exchanges)
            if not exchanges_data:
                logger.warning(f"No exchange data returned for dataset {dataset_id}")
                return False
                
            # Collect all records for bulk insert
            exchanges_to_insert = []
            properties_to_insert = []
            processed = 0
            errors = 0
            
            # Process all exchange types
            for exchange_type, exchanges in exchanges_data.items():
                if not isinstance(exchanges, list):
                    continue
                    
                for exch_data in exchanges:
                    try:
                        # Create exchange record
                        exchange = Exchange(
                            dataset_id=dataset_id,
                            exchange_name=exch_data['name'],
                            exchange_group=exch_data.get('group', ''),
                            comment=exch_data.get('comment', ''),
                            amount=exch_data.get('amount', 0.0),
                            unit=exch_data.get('unit'),
                            uncertainty_type=exch_data.get('uncertainty', {}).get('type') if exch_data.get('uncertainty') else None,
                            uncertainty_pedigree_matrix=exch_data.get('uncertainty', {}).get('pedigreeMatrix') if exch_data.get('uncertainty') else None
                        )
                        
                        # Handle link data if present
                        link = exch_data.get('link', {})
                        if link:
                            exchange.spold_id = link.get('url')
                            exchange.geography = link.get('geography')
                        
                        exchanges_to_insert.append(exchange)
                        
                        # Add properties to the list
                        for prop in exch_data.get('property', []):
                            if not isinstance(prop, dict):
                                continue
                                
                            properties_to_insert.append({
                                'exchange': exchange,  # SQLAlchemy will handle the relationship
                                'property_name': prop.get('name', ''),
                                'unit': prop.get('unit'),
                                'amount': prop.get('amount'),
                                'comment': prop.get('comment', '')
                            })
                        
                        processed += 1
                        
                    except Exception as e:
                        errors += 1
                        logger.error(f"Error processing individual exchange: {str(e)}")
                        continue
            
            if exchanges_to_insert:
                # Bulk insert exchanges
                session.bulk_save_objects(exchanges_to_insert)
                session.flush()
                
                # Now bulk insert properties
                session.bulk_insert_mappings(ExchangeProperty, properties_to_insert)
                
                # Mark as complete
                dataset = session.query(Dataset).get(dataset_id)
                if dataset:
                    dataset.exchanges_complete = True
                
                session.commit()
                
            logger.info(f"Dataset {dataset_id}: Processed {processed} exchanges with {errors} errors")
            return True if processed > 0 else False
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error in bulk processing: {str(e)}")
            return False
        finally:
            session.close()
    def process_dataset_lci(self, ep: ei.EcoinventProcess, dataset_id: int):
        """Process LCI results for a dataset."""
        session = self.Session()
        try:
            # Get LCI results
            lci_results = self.rate_limited_api_call(ep.get_lci)
            if not lci_results:
                return False
                
            # Add results
            for result in lci_results.get('lci_results', []):
                lci_result = LCIResult(
                    dataset_id=dataset_id,
                    substance_name=result['name'],
                    amount=result['amount'],
                    compartment=result.get('compartment'),
                    sub_compartment=result.get('sub_compartment'),
                    unit=result.get('unit'),
                    score=result.get('score'),
                    relative_contribution=result.get('relative_contribution')
                )
                session.add(lci_result)
                
            # Mark LCI as complete    
            dataset = session.query(Dataset).get(dataset_id)
            dataset.lci_complete = True
            
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error processing LCI results for dataset {dataset_id}: {str(e)}")
            return False
        finally:
            session.close()

    def insert_related_datasets(self, dataset_id: int, related: list):
        """Insert related datasets."""
        session = self.Session()
        try:
            for rel_data in related:
                related = RelatedDataset(
                    dataset_id=dataset_id,
                    spold_id=rel_data['spold_id'],
                    version=rel_data['version'],
                    system_model=rel_data['system_model'],
                    description=rel_data['description'],
                    related_dataset_type=rel_data['related_dataset_type'],
                    is_current=rel_data['is_current']
                )
                session.add(related)
                
            dataset = session.query(Dataset).get(dataset_id)
            dataset.related_datasets_complete = True
                
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting related datasets for {dataset_id}: {str(e)}")
            return False
        finally:
            session.close()
def main():
    parser = argparse.ArgumentParser(description='Create ecoinvent database')
    parser.add_argument('--limit', type=int, help='Limit number of datasets to process')
    parser.add_argument('--start-from', type=int, help='Start processing from this dataset ID')
    parser.add_argument('--add-basics', action='store_true', help='Add basic info and impacts')
    parser.add_argument('--add-exchanges', action='store_true', help='Add exchanges')
    parser.add_argument('--add-lci', action='store_true', help='Add LCI results')
    parser.add_argument('--show-status', type=int, help='Show status for dataset ID')
    args = parser.parse_args()

    # Initialize interface
    try:
        my_settings = ei.Settings()
        ep = ei.EcoinventProcess(my_settings)
        ep.set_release(version="3.10.1", system_model="EN15804")
    except Exception as e:
        logger.error(f"Failed to initialize ecoinvent interface: {str(e)}")
        return

    # Create database
    db_creator = EcoinventDatabaseCreator("ecoinvent_3.10.1_EN15804.db")
    
    try:
        # Show status if requested
        if args.show_status:
            session = db_creator.Session()
            dataset = session.query(Dataset).get(args.show_status)
            if dataset:
                print(f"\nStatus for dataset {args.show_status}:")
                print(f"  Basic info: {'Complete' if dataset.basic_info_complete else 'Incomplete'}")
                print(f"  Documentation: {'Complete' if dataset.documentation_complete else 'Incomplete'}")
                print(f"  Exchanges: {'Complete' if dataset.exchanges_complete else 'Incomplete'}")
                print(f"  Impacts: {'Complete' if dataset.impacts_complete else 'Incomplete'}")
                print(f"  LCI: {'Complete' if dataset.lci_complete else 'Incomplete'}")
            else:
                print(f"\nDataset {args.show_status} not found in database")
            session.close()
            return

        # Initialize methods and categories once
        db_creator.initialize_methods_and_categories(ep)
        
        if args.add_basics:
            # Load all datasets from mapping file
            with open('./ecoinvent_interface/data/mappings/3.10_EN15804.json', 'r') as f:
                all_datasets = json.load(f)
            
            # Create initial dataset entries if they don't exist
            logger.info("Ensuring all datasets exist in database...")
            session = db_creator.Session()
            for dataset in tqdm(all_datasets, desc="Creating dataset entries"):
                db_creator.ensure_dataset_exists(
                    dataset_id=dataset['index'],
                    version="3.10.1",
                    system_model="EN15804"
                )
            session.close()
            
            # Get datasets missing basic info as a list
            incomplete = db_creator.get_datasets_missing_data('basic_info')
            
            # Apply start-from filter if specified
            if args.start_from:
                incomplete = [d for d in incomplete if d >= args.start_from]
            
            # Apply limit if specified
            if args.limit:
                incomplete = incomplete[:args.limit]
            
            if not incomplete:
                logger.info("No datasets missing basic info found")
                return
                
            logger.info(f"Found {len(incomplete)} datasets missing basic info")
            
            # Process each dataset
            for dataset_id in tqdm(incomplete, desc="Processing basic info"):
                try:
                    ep.select_process(dataset_id=str(dataset_id))
                    db_creator.process_dataset_basics(ep, dataset_id)
                    sleep(1.5)
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset_id}: {str(e)}")
                    continue
                    
        elif args.add_exchanges:
            # Get all datasets missing exchanges as a list
            datasets = db_creator.get_datasets_missing_data('exchanges')
            
            # Apply start-from filter if specified
            if args.start_from:
                datasets = [d for d in datasets if d >= args.start_from]
            
            # Apply limit if specified
            if args.limit:
                datasets = datasets[:args.limit]
            
            if not datasets:
                logger.info("No datasets missing exchanges found")
                return
                
            logger.info(f"Found {len(datasets)} datasets missing exchanges")
            
            for dataset_id in tqdm(datasets, desc="Adding exchanges"):
                try:
                    ep.select_process(dataset_id=str(dataset_id))
                    success = db_creator.process_dataset_exchanges(ep, dataset_id)
                    if success:
                        logger.info(f"Successfully added exchanges for dataset {dataset_id}")
                    else:
                        logger.warning(f"Failed to add exchanges for dataset {dataset_id}")
                    sleep(1.5)
                except Exception as e:
                    logger.error(f"Error adding exchanges for dataset {dataset_id}: {str(e)}")
                    continue
                    
        elif args.add_lci:
            # Get all datasets missing LCI as a list
            datasets = db_creator.get_datasets_missing_data('lci')
            
            # Apply start-from filter if specified
            if args.start_from:
                datasets = [d for d in datasets if d >= args.start_from]
            
            # Apply limit if specified
            if args.limit:
                datasets = datasets[:args.limit]
            
            if not datasets:
                logger.info("No datasets missing LCI results found")
                return
                
            logger.info(f"Found {len(datasets)} datasets missing LCI results")
            
            for dataset_id in tqdm(datasets, desc="Adding LCI results"):
                try:
                    ep.select_process(dataset_id=str(dataset_id))
                    success = db_creator.process_dataset_lci(ep, dataset_id)
                    if success:
                        logger.info(f"Successfully added LCI results for dataset {dataset_id}")
                    else:
                        logger.warning(f"Failed to add LCI results for dataset {dataset_id}")
                    sleep(1.5)
                except Exception as e:
                    logger.error(f"Error adding LCI results for dataset {dataset_id}: {str(e)}")
                    continue
        
        else:
            logger.info("Please specify what data to add using one of the --add-* options")
            logger.info("Available options:")
            logger.info("  --add-basics: Add basic info and impacts")
            logger.info("  --add-exchanges: Add exchanges")
            logger.info("  --add-lci: Add LCI results")
            logger.info("  --show-status DATASET_ID: Show status for a specific dataset")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    
    logger.info("Database creation process finished!")

if __name__ == "__main__":
    main()