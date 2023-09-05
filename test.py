from google.cloud import storage


client = storage.Client()
bucket = client.get_bucket("ml-pipeline-arxiv-paper-artifact")
blobs = bucket.list_blobs()
for blob in blobs:
    print(blob.name)
