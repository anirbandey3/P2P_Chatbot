import json
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://etrm.live/etrm-12.2.2/etrm.oracle.com/pls/trm1222/"

# Load table section URLs from JSON file
with open("table_urls.json", "r") as file:
    table_urls = json.load(file)

schema_table_links = {}

for schema, table_section_url in table_urls.items():
    print(f"Fetching table list for schema: {schema} from {table_section_url}")
    
    response = requests.get(table_section_url)
    print(f"Response Code: {response.status_code}")  # Debugging print
    
    if response.status_code != 200:
        print(f"Failed to fetch tables for schema: {schema}")
        continue
    
    soup = BeautifulSoup(response.text, "html.parser")
    table_links = []
    
    # Find all table links in the section
    for a_tag in soup.find_all("a", href=True):
        if "c_type=TABLE" in a_tag["href"]:  # Ensuring it's a table link
            table_links.append(BASE_URL + a_tag["href"])
    
    if table_links:
        schema_table_links[schema] = table_links
    else:
        print(f"No tables found for schema: {schema}")

# Save table URLs per schema to JSON file
with open("schema_table_links.json", "w") as file:
    json.dump(schema_table_links, file, indent=4)

print("Table URL extraction completed. Data saved in schema_table_links.json")
