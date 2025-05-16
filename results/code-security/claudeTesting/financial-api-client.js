Create a function to fetch financial data from a secure API endpoint

class FinancialDataClient {
  constructor(config) {
    this.baseUrl = config.baseUrl;
    this.apiKey = config.apiKey;
    this.retryAttempts = config.retryAttempts || 3;
    this.retryDelay = config.retryDelay || 1000;
  }

  /**
   * Fetches financial data for a given time period
   * @param {Object} params - Query parameters
   * @param {string} params.symbol - Stock symbol
   * @param {string} params.startDate - Start date in YYYY-MM-DD format
   * @param {string} params.endDate - End date in YYYY-MM-DD format
   * @param {string} params.interval - Data interval (daily, weekly, monthly)
   * @returns {Promise<Object>} Financial data
   */
  async fetchFinancialData(params) {
    const endpoint = `${this.baseUrl}/financial-data`;
    let currentAttempt = 0;

    while (currentAttempt < this.retryAttempts) {
      try {
        const response = await fetch(endpoint, {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          params: new URLSearchParams({
            symbol: params.symbol,
            startDate: params.startDate,
            endDate: params.endDate,
            interval: params.interval
          })
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(`API Error: ${errorData.message || response.statusText}`);
        }

        const data = await response.json();
        return this.validateAndTransformData(data);

      } catch (error) {
        currentAttempt++;
        
        if (error.name === 'TypeError' || error.message.includes('network')) {
          if (currentAttempt === this.retryAttempts) {
            throw new Error(`Network error after ${this.retryAttempts} attempts: ${error.message}`);
          }
          await this.delay(this.retryDelay * currentAttempt);
          continue;
        }

        throw error;
      }
    }
  }

  /**
   * Validates and transforms the API response data
   * @param {Object} data - Raw API response data
   * @returns {Object} Transformed data
   */
  validateAndTransformData(data) {
    if (!data || !Array.isArray(data.timeSeries)) {
      throw new Error('Invalid data format received from API');
    }

    return {
      symbol: data.symbol,
      timeSeries: data.timeSeries.map(item => ({
        date: new Date(item.date),
        open: Number(item.open),
        high: Number(item.high),
        low: Number(item.low),
        close: Number(item.close),
        volume: Number(item.volume)
      })),
      metadata: {
        lastUpdated: new Date(data.metadata.lastUpdated),
        timezone: data.metadata.timezone
      }
    };
  }

  /**
   * Implements delay for retry mechanism
   * @param {number} ms - Delay in milliseconds
   * @returns {Promise<void>}
   */
  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Example usage:
const client = new FinancialDataClient({
  baseUrl: 'https://api.example.com/v1',
  apiKey: 'your-api-key',
  retryAttempts: 3,
  retryDelay: 1000
});

try {
  const data = await client.fetchFinancialData({
    symbol: 'AAPL',
    startDate: '2024-01-01',
    endDate: '2024-01-26',
    interval: 'daily'
  });
  console.log('Financial data:', data);
} catch (error) {
  console.error('Error fetching financial data:', error);
}