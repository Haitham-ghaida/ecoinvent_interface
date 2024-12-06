import json

# Read the JSON file
with open('./ecoinvent_interface/data/mappings/3.10_EN15804.json', 'r') as f:
   data = json.load(f)

min_entry = min(data, key=lambda x: x['index'])
max_entry = max(data, key=lambda x: x['index'])

print(f"Min index: {min_entry['index']} - {min_entry['activity_name']}")
print(f"Max index: {max_entry['index']} - {max_entry['activity_name']}")