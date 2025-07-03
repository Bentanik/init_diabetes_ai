from io import BytesIO
from minio import Minio
from config import settings


class MinioClient:
    def __init__(self):
        minio_config = settings.get_minio_config()
        self.client = Minio(
            endpoint=minio_config["endpoint"],
            access_key=minio_config["access_key"],
            secret_key=minio_config["secret_key"],
            secure=minio_config["secure"],
        )

    def create_bucket_if_not_exists(self, bucket_name: str):
        """Create a bucket if it doesn't exist"""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
        except Exception as e:
            raise Exception(f"Failed to create bucket: {str(e)}")

    def upload_file(
        self, bucket_name: str, object_name: str, file_data: bytes, content_type: str
    ):
        """Upload a file to MinIO"""
        try:
            self.create_bucket_if_not_exists(bucket_name)
            file_data_io = BytesIO(file_data)
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=file_data_io,
                length=len(file_data),
                content_type=content_type,
            )
            return True
        except Exception as e:
            raise Exception(f"Failed to upload file: {str(e)}")

    def get_file(self, bucket_name: str, object_name: str):
        """Get a file from MinIO"""
        try:
            return self.client.get_object(bucket_name, object_name)
        except Exception as e:
            raise Exception(f"Failed to get file: {str(e)}")

    def delete_file(self, bucket_name: str, object_name: str):
        """Delete a file from MinIO"""
        try:
            self.client.remove_object(bucket_name, object_name)
            return True
        except Exception as e:
            raise Exception(f"Failed to delete file: {str(e)}")


minio_client = MinioClient()
