import psycopg2
from psycopg2 import OperationalError, sql

# Hardcoded connection parameters; adjust as needed.
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "FaceRecognition"
DB_USER = "postgres"
DB_PASSWORD = "somil123"

def get_connection():
    """
    Returns a new psycopg2 connection instance if successful, or raises an exception.
    Caller is responsible for closing the connection.
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=5
        )
        return conn
    except OperationalError as e:
        # Print/log the error, then re-raise so caller knows it failed
        print(f"Failed to connect to PostgreSQL: {e}")
        raise

def test_query():
    """
    Simple function to test that the connection works by running a trivial query.
    Returns the PostgreSQL version string.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        cur.close()
        return version
    finally:
        if conn:
            conn.close()

# Example usage when run as script
if __name__ == "__main__":
    try:
        version = test_query()
        print("Connected to PostgreSQL! Version:", version)
    except Exception:
        pass
