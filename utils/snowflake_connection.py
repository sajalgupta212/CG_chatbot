import os
from dotenv import load_dotenv
import snowflake.connector
from cryptography.hazmat.primitives import serialization

load_dotenv()

def get_connection():
    """
    Establishes a Snowflake connection using JWT authentication
    based on environment variables.
    """

    user = os.getenv("SNOWFLAKE_USER")
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")
    role = os.getenv("SNOWFLAKE_ROLE")

    private_key_file = os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE")
    private_key_passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")

    if not user or not account or not private_key_file:
        raise ValueError("Missing required Snowflake configuration variables")

    # Load the RSA private key from the .p8 file
    with open(private_key_file, "rb") as f:
        key_bytes = f.read()

    private_key = serialization.load_pem_private_key(
        key_bytes,
        password=private_key_passphrase.encode() if private_key_passphrase else None
    )

    # Convert private key to DER format for Snowflake
    private_key_der = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    # Connect to Snowflake using JWT
    conn = snowflake.connector.connect(
        user=user,
        account=account,
        authenticator="SNOWFLAKE_JWT",
        private_key=private_key_der,
        warehouse=warehouse,
        database=database,
        schema=schema,
        role=role
    )

    return conn

if __name__ == "__main__":
    # Quick test
    connection = get_connection()
    print("Connected to Snowflake successfully!")
    connection.close()
