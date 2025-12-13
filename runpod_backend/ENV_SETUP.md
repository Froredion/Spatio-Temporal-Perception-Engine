# Environment Setup Guide (Linux VPS)

## Step-by-Step Instructions

### 1. Navigate to the backend directory

```bash
cd /path/to/runpod_backend
```

### 2. Create the .env file

```bash
nano .env
```

### 3. Paste your environment variables

Copy the contents from your local machine's `.env` file and paste them into the editor.

The file should contain these variables:

```
# R2 Account & Bucket
R2_ACCOUNT_ID=your_account_id
R2_BUCKET_NAME=your_bucket_name

# R2 API Credentials (for uploads/writes)
R2_ACCESS_KEY_ID=your_access_key_id
R2_SECRET_ACCESS_KEY=your_secret_access_key

# R2 Public URL (for reads/downloads)
R2_PUBLIC_URL=https://your-public-url.r2.dev

# R2 Endpoint URL (for S3-compatible API)
R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com/your_bucket_name
```

### 4. Save and exit

- Press `Ctrl + O` to save
- Press `Enter` to confirm
- Press `Ctrl + X` to exit

### 5. Verify the file was created

```bash
cat .env
```

### 6. Set proper permissions (recommended)

```bash
chmod 600 .env
```

This restricts the file to be readable only by the owner.

## Alternative: Using echo (quick paste)

If you prefer a one-liner, you can use:

```bash
cat > .env << 'EOF'
# Paste your .env contents here
EOF
```

Then paste your variables and type `EOF` on a new line to finish.

## Verify Environment Variables Load Correctly

```bash
source .env
echo $R2_BUCKET_NAME
```

If it prints your bucket name, the setup is complete.
