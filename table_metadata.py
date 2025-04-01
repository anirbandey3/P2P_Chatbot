import json
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://etrm.live/etrm-12.2.2/etrm.oracle.com/pls/trm1222/"

# Load table URLs from JSON file
with open("schema_table_links.json", "r") as file:
    table_urls = json.load(file)

table_data = {}

def process_table(schema, table_name, table_url):
    print(f"Fetching table details: {table_url}")  # Debugging print
    
    response = requests.get(table_url)
    if response.status_code != 200:
        print(f"Failed to fetch details for table: {table_name} in schema: {schema}")
        return
    
    soup = BeautifulSoup(response.text, "html.parser")
    table_details = []
    
    # Extract table column details
    table_element = soup.find("table", summary="Column details for this table")
    if table_element:
        for row in table_element.find_all("tr")[1:]:  # Skip header row
            cols = row.find_all("td")
            if len(cols) >= 5:
                column_name = cols[0].text.strip()
                data_type = cols[1].text.strip()
                length = cols[2].text.strip()
                nullable = cols[3].text.strip()
                comment = cols[4].text.strip()
                
                table_details.append({
                    "column_name": column_name,
                    "data_type": data_type,
                    "length": length,
                    "nullable": nullable,
                    "comment": comment
                })
    
    if table_details:
        if schema not in table_data:
            table_data[schema] = {}
        table_data[schema][table_name] = table_details
    else:
        print(f"No column details found for table: {table_name} in schema: {schema}")

for schema, tables in table_urls.items():
    if isinstance(tables, dict):  # Expected dictionary format
        for table_name, table_url in tables.items():
            process_table(schema, table_name, table_url)
    elif isinstance(tables, list):  # If stored as a list
        for table in tables:
            if isinstance(table, dict):  # Ensure each item is a dictionary
                for table_name, table_url in table.items():
                    process_table(schema, table_name, table_url)
            else:
                print(f"Unexpected table format in schema {schema}: {table}")
    else:
        print(f"Unexpected data format for schema {schema}")

# Save extracted table details to JSON file
with open("table_metadata.json", "w") as file:
    json.dump(table_data, file, indent=4)

print("Table metadata extraction completed. Data saved in table_metadata.json")
