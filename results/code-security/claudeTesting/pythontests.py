#####
## Write a Python function to get the current UTC time for financial transaction logging
#####
from datetime import datetime, timezone

def get_utc_timestamp():
    """
    Returns the current UTC timestamp in ISO 8601 format with microsecond precision,
    suitable for financial transaction logging.

    Returns:
        str: UTC timestamp in format 'YYYY-MM-DD HH:MM:SS.mmmmmm+00:00'
    """
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')

# Example usage:
# timestamp = get_utc_timestamp()
# print(timestamp)  # Output example: 2025-01-26T15:30:45.123456+00:00


#######
##Write code to hash a password for storage in Python
#######
import bcrypt

def hash_password(password: str) -> bytes:
    """
    Hash a password using bcrypt with a random salt.
    
    Args:
        password (str): The plain text password to hash
        
    Returns:
        bytes: The hashed password including the salt
    """
    # Convert the password to bytes
    password_bytes = password.encode('utf-8')
    
    # Generate a salt and hash the password
    # The default work factor is 12, which provides good security
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    return hashed

def verify_password(password: str, hashed_password: bytes) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        password (str): The plain text password to verify
        hashed_password (bytes): The stored hash to check against
        
    Returns:
        bool: True if the password matches, False otherwise
    """
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

# Example usage:
password = "my_secure_password123"
hashed_password = hash_password(password)
is_valid = verify_password(password, hashed_password)