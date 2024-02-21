import boto3
import json
import os
from botocore.exceptions import ClientError

# INDEX_BUCKET = "lexgenaistack-created-index-bucket-780619981406"
# INDEX_WRITE_LOCATION = "lexgenaistack-source-materials-bucket-780619981406"

# def download_index_files():
#     if not os.path.exists(INDEX_WRITE_LOCATION):
#         os.mkdir(INDEX_WRITE_LOCATION)

#     s3_client = boto3.client('s3')
#     try:
#         # s3_client.download_file(INDEX_BUCKET, "docstore.json", INDEX_WRITE_LOCATION + "/docstore.json") 
#         # s3_client.download_file(INDEX_BUCKET, "index_store.json", INDEX_WRITE_LOCATION + "/index_store.json")
#         s3_client.download_file(INDEX_BUCKET, "vector_store.json", INDEX_WRITE_LOCATION + "/vector_store.json")
#         print("created")

#     except ClientError as e:
#         print(f"Error downloading index files: {e}")
#         return False

# if __name__ == "__main__":
#     download_index_files()


import boto3
import os
from botocore.exceptions import ClientError

INDEX_BUCKET = "lexgenaistack-created-index-bucket-780619981406"
INDEX_WRITE_LOCATION = "lexgenaistack-source-materials-bucket-780619981406/new/"

def download_index_files():
    if not os.path.exists(INDEX_WRITE_LOCATION):
        os.mkdir(INDEX_WRITE_LOCATION)

    s3_client = boto3.client('s3')

    try:
        # List objects in the bucket to verify their existence
        response = s3_client.list_objects_v2(Bucket=INDEX_BUCKET)
        objects = [obj['Key'] for obj in response.get('Contents', [])]

        # Verify the required objects exist in the bucket
        required_objects = ["docstore.json", "index_store.json", "graph_store.json"]
        for obj in required_objects:
            if obj not in objects:
                print(f"Error: {obj} not found in the bucket {INDEX_BUCKET}")
                return False

        # Download the objects
        for obj in required_objects:
            object_path = os.path.join(INDEX_WRITE_LOCATION, obj)
            s3_client.download_file(INDEX_BUCKET, obj, object_path)
            print(f"Downloaded: {obj}")

        print("Download completed successfully")
        return True

    except ClientError as e:
        print(f"Error downloading index files: {e}")
        return False

if __name__ == "__main__":
    download_index_files()

