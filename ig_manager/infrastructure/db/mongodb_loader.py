import csv
import pymongo
from datetime import datetime, timedelta

# MongoDB configuration
mongo_client = pymongo.MongoClient("mongodb://root:passwd@mongo:27017/")
db = mongo_client["car_posts"]
collection = db["posts"]

# Parse CSV file and insert data into MongoDB
csv_file_path = 'cars.csv'

# Start scheduling from today
schedule_start_date = datetime.now()


with open(csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for idx, row in enumerate(reader):
        # Split the name field into name, description, and tech_data
        name_parts = row['name'].split(',"')
        if len(name_parts) == 3:
            name = name_parts[0].strip('"')
            description = name_parts[1].strip('"')
            tech_data = name_parts[2].strip('"')
        else:
            name = row['name']
            description = ''
            tech_data = ''

        # Calculate the schedule date for each entry
        schedule_date = schedule_start_date + timedelta(days=idx)

        # Create a new document with the split fields and schedule date
        document = {
            'name': name,
            'description': description,
            'tech_data': tech_data,
            'schedule_date': schedule_date.isoformat(),
            'approved': False  # Initially not approved
        }

        collection.insert_one(document)

print("Data loaded successfully into MongoDB.")