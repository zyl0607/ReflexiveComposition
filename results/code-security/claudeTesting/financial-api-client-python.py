Create a function to fetch financial data from a secure API endpoint

import requests
from typing import Dict, Any, Optional
from datetime import datetime
import time
from dataclasses import dataclass
from urllib.parse import urljoin

@dataclass
class APIConfig:
    base_url: str
    api_key: str
    retry_attempts: int = 3
    retry_delay: float = 1.0

class FinancialDataClient:
    def __init__(self, config: APIConfig):
        """
        Initialize the Financial Data API client
        
        Args:
            config: APIConfig object containing base_url, api_key, and retry settings
        """
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.retry_attempts = config.retry_attempts
        self.retry_delay = config.retry_delay
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def fetch_financial_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = 'daily'
    ) -> Dict[str, Any]:
        """
        Fetch financial data for a given time period
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (daily, weekly, monthly)
            
        Returns:
            Dict containing the financial data
            
        Raises:
            RequestError: If the API request fails
            ValidationError: If the response data is invalid
        """
        endpoint = urljoin(self.base_url, 'financial-data')
        params = {
            'symbol': symbol,
            'startDate': start_date,
            'endDate': end_date,
            'interval': interval
        }

        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(endpoint, params=params)
                response.raise_for_status()
                
                data = response.json()
                return self._validate_and_transform_data(data)
                
            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.ConnectionError):
                    if attempt == self.retry_attempts - 1:
                        raise RequestError(
                            f"Network error after {self.retry_attempts} attempts: {str(e)}"
                        ) from e
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                
                raise RequestError(f"API request failed: {str(e)}") from e

    def _validate_and_transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and transform the API response data
        
        Args:
            data: Raw API response data
            
        Returns:
            Transformed data dictionary
            
        Raises:
            ValidationError: If the data format is invalid
        """
        if not data or not isinstance(data.get('timeSeries'), list):
            raise ValidationError('Invalid data format received from API')

        return {
            'symbol': data['symbol'],
            'time_series': [
                {
                    'date': datetime.fromisoformat(item['date']).date(),
                    'open': float(item['open']),
                    'high': float(item['high']),
                    'low': float(item['low']),
                    'close': float(item['close']),
                    'volume': int(item['volume'])
                }
                for item in data['timeSeries']
            ],
            'metadata': {
                'last_updated': datetime.fromisoformat(data['metadata']['lastUpdated']),
                'timezone': data['metadata']['timezone']
            }
        }

class RequestError(Exception):
    """Raised when an API request fails"""
    pass

class ValidationError(Exception):
    """Raised when response data validation fails"""
    pass


# Example usage:
if __name__ == '__main__':
    config = APIConfig(
        base_url='https://api.example.com/v1',
        api_key='your-api-key',
        retry_attempts=3,
        retry_delay=1.0
    )
    
    client = FinancialDataClient(config)
    
    try:
        data = client.fetch_financial_data(
            symbol='AAPL',
            start_date='2024-01-01',
            end_date='2024-01-26',
            interval='daily'
        )
        print('Financial data:', data)
    except (RequestError, ValidationError) as e:
        print('Error fetching financial data:', str(e))