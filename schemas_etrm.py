import requests
from bs4 import BeautifulSoup
import json

# Base URL for schemas
BASE_URL = "https://etrm.live/etrm-12.2.2/etrm.oracle.com/pls/trm1222/"
MAIN_URL = BASE_URL + "etrm_pnav91de.html"

# Step 1: Fetch schemas
def get_schemas():
    response = requests.get(MAIN_URL)
    if response.status_code != 200:
        print("Failed to fetch schemas")
        return {}
    
    soup = BeautifulSoup(response.text, "html.parser")
    schemas = {}
    
    for dt in soup.find_all("dt"):  # Finding schema links
        a_tag = dt.find("a")
        if a_tag and "c_owner" in a_tag["href"]:
            schema_name = a_tag.text.strip()
            schema_url = BASE_URL + a_tag["href"]
            schemas[schema_name] = schema_url
    
    return schemas

schemas = get_schemas()

# Save schemas to JSON
with open("schemas.json", "w") as f:
    json.dump(schemas, f, indent=4)

print("Schemas fetched and saved successfully!")
