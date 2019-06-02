import os
import boto3
from dotenv import load_dotenv
load_dotenv()

DO_ACCESS_KEY_ID = os.getenv("DO_ACCESS_KEY_ID")
DO_SECRET_ACCESS_KEY = os.getenv("DO_SECRET_ACCESS_KEY")

# Digital Ocean client
client = boto3.client(
            's3',
            region_name='ams3',
            endpoint_url='https://ams3.digitaloceanspaces.com',
            aws_access_key_id=DO_ACCESS_KEY_ID,
            aws_secret_access_key=DO_SECRET_ACCESS_KEY
        )

xy_names = ['in', 'out']

# List outer folders (train, test, val) e.g.
resp = client.list_objects(Bucket='illumination-mobile', Delimiter='/')
resp_cont = resp['CommonPrefixes']
data_folders = [r['Prefix'] for r in resp_cont]

# Create matching local folder structure
for f in data_folders:
    for n in xy_names:
        if not os.path.isdir(f'data/{f}{n}'):
            os.makedirs(f'data/{f}{n}')

for folder in data_folders:
    # Download from x and y folders
    prefixes = [f'data/{folder}{n}' for n in xy_names]

    for prefix in prefixes:     
        # Get image names
        resp = client.list_objects(Bucket='illumination-mobile', Prefix=prefix)
        resp_cont = resp['Contents']
        image_names = [r['Key'] for r in resp_cont]
        # Exclude folder name
        image_names = image_names[1:]

        for name in image_names:
            # Download file
            resp = client.download_file('illumination-mobile', name, name)

