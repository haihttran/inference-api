import boto3
import yaml
import os, sys
import time

script_dir = os.path.dirname(__file__)
with open(os.path.join(script_dir,"setting.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
#provided Access Credentialsaccess_key
# access_key = "AKIA2SHQJEZHPPEYNT6N"
# secret_key = "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"
# access_key = config['ACCESS_KEY']
# secret_key = config['SECRET_KEY']
BUCKET_NAME = config['BUCKET_NAME']
BUCKET_PATH = config['PATH']
DEST_BUCKET = config['UPLOAD_BUCKET']

""" Establish aws boto3 session
    and access to s3 resources """
session = boto3.Session(
    aws_access_key_id = config['ACCESS_KEY'],
    aws_secret_access_key = config['SECRET_KEY']
)
s3 = session.resource('s3')

def download_s3_file(s3_file_name):
    bucket_file_name = BUCKET_PATH + s3_file_name
    # temp = 'swine_videos/Loading_Unloading/KE_CH_0856.mp4'
    local_file_name = os.path.join(os.path.join(os.getcwd(),'s3_downloads'), s3_file_name)
    my_bucket = s3.Bucket(BUCKET_NAME)
    bucket_file_name = BUCKET_PATH + s3_file_name
    # print(bucket_file_name)
    # my_bucket.download_file(temp, local_file_name)
    my_bucket.download_file(bucket_file_name, local_file_name)
    return local_file_name


def upload_s3_file(local_file_path, local_file_name):
    path = 'uploads/' + local_file_name
    s3.meta.client.upload_file(local_file_path, DEST_BUCKET, path)
    return "s3://{}/{}".format(DEST_BUCKET, path)

if __name__ == '__main__':
    print(os.path.join(os.path.join(os.getcwd(),'s3_downloads'), 'KE_CH1_3020_zoom.mp4'))
    print(BUCKET_PATH + 'CE_Ch5_1418-43.mp4')
    # print(os.path.join(os.path.dirname(script_dir),'s3_downloads'))