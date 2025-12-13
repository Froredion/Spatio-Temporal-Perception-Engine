"""
Cloudflare R2 Storage Client

S3-compatible object storage for video files, frames, and embeddings.
"""

import os
import boto3
from botocore.config import Config
from typing import Optional, BinaryIO
from io import BytesIO


class R2StorageClient:
    """Cloudflare R2 storage client using S3-compatible API."""

    def __init__(self):
        self.account_id = os.environ.get('R2_ACCOUNT_ID')
        self.bucket_name = os.environ.get('R2_BUCKET_NAME')
        self.access_key = os.environ.get('R2_ACCESS_KEY_ID')
        self.secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')
        self.public_url = os.environ.get('R2_PUBLIC_URL', '').rstrip('/')
        endpoint_url_raw = os.environ.get('R2_ENDPOINT_URL', '')

        # Parse endpoint URL - it may include bucket name as path
        # e.g., https://account.r2.cloudflarestorage.com/bucket-name
        from urllib.parse import urlparse
        if endpoint_url_raw:
            parsed = urlparse(endpoint_url_raw)
            # Extract base endpoint (scheme + netloc)
            self.endpoint_url = f"{parsed.scheme}://{parsed.netloc}"
            # If path contains bucket name, use it (overrides R2_BUCKET_NAME)
            if parsed.path and parsed.path.strip('/'):
                path_bucket = parsed.path.strip('/')
                if path_bucket:
                    print(f"[R2] Using bucket from endpoint URL path: {path_bucket}")
                    self.bucket_name = path_bucket
        else:
            self.endpoint_url = None

        if not self.endpoint_url and self.account_id:
            self.endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"

        print(f"[R2] Initialized - endpoint: {self.endpoint_url}, bucket: {self.bucket_name}")
        self._client = None

    @property
    def client(self):
        """Lazy-load the S3 client."""
        if self._client is None:
            if not all([self.access_key, self.secret_key, self.endpoint_url]):
                raise ValueError("R2 credentials not configured. Set R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, and R2_ACCOUNT_ID")

            self._client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(
                    signature_version='s3v4',
                    s3={'addressing_style': 'path'}
                ),
            )
        return self._client

    def upload_file(
        self,
        file_path: str,
        key: str,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload a local file to R2.

        Args:
            file_path: Local file path
            key: Object key in bucket (e.g., 'videos/raw/abc123.mp4')
            content_type: MIME type (auto-detected if not provided)

        Returns:
            Public URL of the uploaded file
        """
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type

        self.client.upload_file(
            file_path,
            self.bucket_name,
            key,
            ExtraArgs=extra_args,
        )

        return f"{self.public_url}/{key}"

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload bytes data to R2.

        Args:
            data: Bytes to upload
            key: Object key in bucket
            content_type: MIME type

        Returns:
            Public URL of the uploaded file
        """
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type

        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
            **extra_args,
        )

        return f"{self.public_url}/{key}"

    def download_file(self, key: str, local_path: str) -> str:
        """
        Download a file from R2 to local path.

        Args:
            key: Object key in bucket
            local_path: Local destination path

        Returns:
            Local file path
        """
        self.client.download_file(self.bucket_name, key, local_path)
        return local_path

    def download_bytes(self, key: str) -> bytes:
        """
        Download a file from R2 as bytes.

        Args:
            key: Object key in bucket

        Returns:
            File contents as bytes
        """
        response = self.client.get_object(Bucket=self.bucket_name, Key=key)
        return response['Body'].read()

    def get_public_url(self, key: str) -> str:
        """
        Get the public URL for an object.

        Args:
            key: Object key in bucket

        Returns:
            Public URL
        """
        return f"{self.public_url}/{key}"

    def file_exists(self, key: str) -> bool:
        """Check if a file exists in R2."""
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except:
            return False

    def list_files(self, prefix: str = '', sort_by_modified: bool = False) -> list:
        """
        List files in R2 with optional prefix filter.
        Handles pagination for large result sets.

        Args:
            prefix: Key prefix to filter by
            sort_by_modified: If True, sort by LastModified (newest first)

        Returns:
            List of object keys (sorted by LastModified if requested)
        """
        try:
            all_objects = []
            continuation_token = None

            print(f"[R2] Listing files with prefix: '{prefix}' in bucket: '{self.bucket_name}'")

            while True:
                kwargs = {
                    'Bucket': self.bucket_name,
                    'Prefix': prefix,
                }
                if continuation_token:
                    kwargs['ContinuationToken'] = continuation_token

                response = self.client.list_objects_v2(**kwargs)

                print(f"[R2] list_objects_v2 response keys: {list(response.keys())}")

                if 'Contents' in response:
                    all_objects.extend(response['Contents'])
                    print(f"[R2] Found {len(response['Contents'])} files in this batch, total: {len(all_objects)}")
                else:
                    print(f"[R2] No 'Contents' in response")

                # Check if there are more results
                if response.get('IsTruncated'):
                    continuation_token = response.get('NextContinuationToken')
                    print(f"[R2] Results truncated, fetching more...")
                else:
                    break

            # Sort by LastModified if requested (newest first)
            if sort_by_modified and all_objects:
                all_objects.sort(key=lambda x: x.get('LastModified', ''), reverse=True)
                print(f"[R2] Sorted {len(all_objects)} files by LastModified (newest first)")

            keys = [obj['Key'] for obj in all_objects]
            print(f"[R2] Total files found: {len(keys)}")
            return keys

        except Exception as e:
            # Handle case where prefix doesn't exist or other errors
            print(f"[R2] list_files error for prefix '{prefix}': {e}")
            import traceback
            traceback.print_exc()
            return []

    def delete_file(self, key: str) -> bool:
        """Delete a file from R2."""
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=key)
            return True
        except:
            return False


# Singleton instance
_r2_client: Optional[R2StorageClient] = None


def get_r2_client() -> R2StorageClient:
    """Get or create the R2 storage client singleton."""
    global _r2_client
    if _r2_client is None:
        _r2_client = R2StorageClient()
    return _r2_client
