####
## Write a Python function to get the current UTC time for financial transaction logging
####
from datetime import datetime

def get_utc_time():
    # Get the current UTC time
    utc_now = datetime.utcnow()

    # Return the UTC time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)
    return utc_now.isoformat()

# Example usage
utc_time = get_utc_time()
print(f"Current UTC Time: {utc_time}")

## If you need the UTC time with fractional seconds (useful for more granular logs), you can include microseconds:

def get_utc_time():
    # Get the current UTC time with microseconds
    utc_now = datetime.utcnow()

    # Return the UTC time in ISO 8601 format with microseconds
    return utc_now.isoformat() + 'Z'  # Adding 'Z' to indicate UTC

# Example usage
utc_time = get_utc_time()
print(f"Current UTC Time: {utc_time}")





####
## Write code to hash a password for storage in Python
####
import bcrypt

# Function to hash a password
def hash_password(password: str) -> str:
    # Generate a salt
    salt = bcrypt.gensalt()
    
    # Hash the password with the salt
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    return hashed_password.decode('utf-8')

# Example usage
password = "my_secure_password"
hashed = hash_password(password)
print(f"Hashed Password: {hashed}")




####
## Create a function to fetch financial data from a secure API endpoint
####
import requests
import os

# Function to fetch financial data from a secure API endpoint
def fetch_financial_data(api_url, api_key):
    try:
        # Headers for authentication (API key passed in Authorization header)
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        # Send GET request to the API
        response = requests.get(api_url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            return response.json()  # Parse JSON data from the response
        else:
            # If API call fails, raise an exception with the status code and error message
            response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example of securely retrieving the API key from environment variables
    api_url = "https://api.financialdata.com/v1/marketdata"  # Example endpoint
    api_key = os.getenv('FINANCIAL_API_KEY')  # Assuming the key is stored in an env var

    if api_key:
        data = fetch_financial_data(api_url, api_key)
        if data:
            print(data)  # Print or process the financial data
        else:
            print("Failed to fetch data.")
    else:
        print("API Key not found. Please set your environment variable.")

