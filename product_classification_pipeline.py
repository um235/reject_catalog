"""
Product extraction, filtering, and classification pipeline.
This script handles:
1. Extracting product data and inserting into the database
2. Creating and saving filters
3. Classifying products against those filters
"""

import logging
from typing import List, Dict, Any

# Import from existing project files
from ai import extract_pairs, plan_from_query, embed, cosine
from db import DB
from config import DATABASE_PATH
from logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def extract_and_store_products(db: DB, products: List[Dict[str, str]]):
    """Extract characteristics from products and store them in the database"""
    logger.info(f"Extracting and storing {len(products)} products")

    for product in products:
        # Insert basic product info
        db.upsert_item(product["id"], product["title"], product["description"])

        # Extract material characteristics using LLM
        pairs = extract_pairs(product["title"], product["description"])
        logger.info(f"Extracted {len(pairs)} characteristics from product {product['id']}")

        # Insert extracted occurrences
        occurrences = []
        for pair in pairs:
            occurrences.append((
                product["id"],
                pair.get("name", ""),
                pair.get("value", ""),
                pair.get("unit", ""),
                pair.get("number", 0.0),
                f"{product['title']} - {product['description']}"
            ))

        if occurrences:
            db.add_occurrences(occurrences)


def create_filters(db: DB, filter_definitions: List[Dict[str, str]]):
    """Create and save filters based on natural language queries"""
    logger.info(f"Creating {len(filter_definitions)} filters")

    for filter_def in filter_definitions:
        logger.info(f"Creating filter: {filter_def['name']}")
        filter_plan = plan_from_query(filter_def["query"])
        db.save_filter(filter_def["id"], filter_def["name"], filter_plan)
        logger.info(f"Filter plan: {filter_plan}")


def classify_products(db: DB):
    """Classify products against all filters"""
    logger.info("Starting product classification")

    # Load all filters
    filters = db.load_filters()
    logger.info(f"Loaded {len(filters)} filters")

    # Load all items
    items = db.fetch_items()
    logger.info(f"Loaded {len(items)} products")

    # Results will store product_id -> [matching_filter_ids]
    classification_results = {}

    # For each item, check which filters it matches
    for item in items:
        item_id = item["item_id"]
        title = item["title"]
        occurrences = db.fetch_occ_by_item(item_id)

        logger.info(f"Classifying product {item_id}: {title}")
        matching_filters = []

        for filter_def in filters:
            filter_id = filter_def["filter_id"]
            filter_name = filter_def["name"]
            filter_plan = filter_def["plan"]

            if matches_filter(item, occurrences, filter_plan):
                matching_filters.append(filter_id)
                logger.info(f"  - Matches filter '{filter_name}' ({filter_id})")
            else:
                logger.debug(f"  - Does not match filter '{filter_name}' ({filter_id})")

        classification_results[item_id] = matching_filters

    return classification_results


def matches_filter(item, occurrences, filter_plan):
    """Determine if an item matches a given filter plan"""
    # Create a dictionary of attribute names to values for easy lookup
    item_attrs = {}
    for occ in occurrences:
        name = occ["name_text"]
        if name not in item_attrs:
            item_attrs[name] = []
        item_attrs[name].append({
            "value": occ["value_text"],
            "unit": occ["unit_text"],
            "number": occ["number_value"]
        })

    # Check must conditions (all must match)
    for condition in filter_plan.get("must", []):
        attr = condition.get("attr")
        op = condition.get("op")
        value = condition.get("value")
        unit = condition.get("unit")

        if attr not in item_attrs:
            return False

        match_found = False
        for attr_value in item_attrs[attr]:
            if op == "eq" and str(attr_value["value"]) == str(value):
                match_found = True
                break
            elif op == "neq" and str(attr_value["value"]) != str(value):
                match_found = True
                break
            elif op == "in" and str(value) in str(attr_value["value"]):
                match_found = True
                break
            elif op == "contains" and str(attr_value["value"]) in str(value):
                match_found = True
                break
            elif attr_value["number"] is not None:
                if op == "gte" and attr_value["number"] >= float(value):
                    match_found = True
                    break
                elif op == "lte" and attr_value["number"] <= float(value):
                    match_found = True
                    break
                elif op == "range" and isinstance(value, list) and len(value) == 2:
                    if attr_value["number"] >= float(value[0]) and attr_value["number"] <= float(value[1]):
                        match_found = True
                        break

        if not match_found:
            return False

    # Check must_not conditions (none should match)
    for condition in filter_plan.get("must_not", []):
        attr = condition.get("attr")
        if attr not in item_attrs:
            continue

        op = condition.get("op")
        value = condition.get("value")

        for attr_value in item_attrs[attr]:
            match_found = False
            if op == "eq" and str(attr_value["value"]) == str(value):
                return False
            elif op == "in" and str(value) in str(attr_value["value"]):
                return False
            # Add more operators as needed

    # If we get here, all must conditions matched and no must_not conditions matched
    # Should conditions could be implemented as a scoring mechanism if needed
    return True


def main():
    """Main function to run the entire pipeline"""
    logger.info("Starting data processing pipeline")

    # Initialize database connection
    db = DB(DATABASE_PATH)

    # Step 1: Sample data - in a real scenario, you would load this from an external source
    products = [
        {
            "id": "prod0011",
            "title": "Кабель ВВГнг(А)-LS 3х2.5 мм²",
            "description": "Силовой кабель с медными жилами, негорючий, с низким дымо- и газовыделением. ГОСТ 31996-2012"
        },
        {
            "id": "prod0021",
            "title": "Провод ПВС 3х1.5 мм² 450/750 В",
            "description": "Гибкий провод с ПВХ изоляцией для бытовых электроприборов"
        },
        {
            "id": "prod0031",
            "title": "Кабель NYM 5х6 мм²",
            "description": "Силовой кабель с медными жилами и ПВХ изоляцией, не распространяющий горение"
        }
    ]

    # Step 1: Extract and insert to database
    logger.info("STEP 1: Extracting and inserting products to database")
    extract_and_store_products(db, products)

    # Step 2: Create filters
    logger.info("STEP 2: Creating filters")
    filter_definitions = [
        {
            "id": "filter001",
            "name": "Силовые кабели",
            "query": "Силовые кабели с медными жилами и сечением не менее 2.5 мм²"
        },
        {
            "id": "filter002",
            "name": "Негорючие кабели",
            "query": "Негорючие кабели с низким дымовыделением"
        },
        {
            "id": "filter003",
            "name": "ПВХ изоляция",
            "query": "Кабели с ПВХ изоляцией и напряжением 450/750 В"
        }
    ]
    create_filters(db, filter_definitions)

    # Step 3: Classify products
    logger.info("STEP 3: Classifying products by filters")
    classification_results = classify_products(db)

    # Display classification summary
    logger.info("Classification results:")
    for item_id, matching_filters in classification_results.items():
        logger.info(f"Product {item_id} matches {len(matching_filters)} filters: {matching_filters}")

    logger.info("Pipeline completed successfully")

main()