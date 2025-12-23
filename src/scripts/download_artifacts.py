import boto3
import os
from settings import Settings


def download_artifacts(settings: Settings):

    s3 = boto3.client('s3')

    os.makedirs('./model/', exist_ok=True)

    s3.download_file(settings.s3_bucket, 'classifier.joblib', './model/classifier.joblib')

    files = s3.list_objects_v2(Bucket=settings.s3_bucket)['Contents']

    for file in files:
        key = file['Key']
        if key.endswith("/"):
            os.makedirs('./model/'+key, exist_ok=True)
        else:
            os.makedirs(os.path.dirname('./model/'+key), exist_ok=True)
            s3.download_file(settings.s3_bucket, key, './model/'+key)