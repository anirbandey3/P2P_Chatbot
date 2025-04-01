import json
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://etrm.live/etrm-12.2.2/etrm.oracle.com/pls/trm1222/"

# Load schemas from JSON file
with open("schemas.json", "r") as file:
    schemas = json.load(file)

table_urls = {}

for schema, relative_url in schemas.items():
    schema_url = relative_url  # Fix applied here
    print(f"Fetching: {schema_url}")  # Debugging print
    
    response = requests.get(schema_url)
    print(f"Response Code: {response.status_code}")  # Debugging print
    
    if response.status_code != 200:
        print(f"Failed to fetch TABLE link for schema: {schema}")
        continue
    
    print(f"Response Content (first 500 chars): {response.text[:500]}")  # Debugging print
    
    soup = BeautifulSoup(response.text, "html.parser")
    table_section = None
    
    # Find the link for TABLE (xxx)
    for dd in soup.find_all("dd"):
        link = dd.find("a")
        if link and "TABLE" in link.text:
            table_section = BASE_URL + link["href"]
            break
    
    if table_section:
        table_urls[schema] = table_section
    else:
        print(f"No TABLE section found for schema: {schema}")

# Save TABLE URLs to JSON file
with open("table_urls.json", "w") as file:
    json.dump(table_urls, file, indent=4)

print("TABLE section URL extraction completed. Data saved in table_urls.json")
