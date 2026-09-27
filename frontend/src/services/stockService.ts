// Live stock data interfaces
export interface StockData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: string;
  lastUpdated: string;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
}

export interface PricePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  price: number; // Added for chart display compatibility
  currency?: string; // Add this field
}

export interface PredictionResult {
  predictedPrice: number;
  currentPrice?: number;  // Current price in same currency as prediction (USD)
  confidence: number;
  model: string;
  timeframe: string;
  priceRange?: [number, number];
  timeFrameDays?: number;
  modelInfo?: {
    model: string;
    members?: string[];
    weights?: Record;
    ensemble_size?: number;
  };
  dataPointsUsed?: number;
  lastUpdated?: string;
  currency?: string;
  dataSource?: string;  // 'live_api' or 'stored_data'
  dataDate?: string | null;  // Date of stored data (null for live_api)
  sourceReliable?: boolean;  // true if real-time, false if historical
}

// Live price response from backend
export interface LivePriceResponse {
  symbol: string;
  price: number;
  timestamp: string;
  source: string;
  source_reliable?: boolean;  // true if real-time, false if historical
  data_date?: string | null;  // Date of data (null for live, date string for permanent fallback)
  company_name: string;
  currency: string;
  exchange_rate?: number;
  exchange_source?: string;
  price_inr?: number;
  price_usd?: number;
  sector?: string;
  market_cap?: string;
  headquarters?: string;
  exchange?: string;
  open?: number;
  high?: number;
  low?: number;
  volume?: number;
  close?: number;
}

// Stock info response from backend (fast metadata)
export interface StockInfoResponse {
  symbol: string;
  company_name: string;
  sector: string;
  market_cap: string;
  headquarters: string;
  exchange: string;
  category: string;
  indicators?: {
    rsi?: number;
    rsi_tag?: string;
    golden_cross?: boolean;
  };
}

// API response wrapper
interface ApiResponse {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Backend API configuration
const BACKEND_BASE_URL = 'http://localhost:5000';
const REQUEST_TIMEOUT = 30000; // 30 seconds

// Cache for storing live price data
const cache = new Map();

// Cache helper functions
function getCachedData(key: string, maxAge: number): T | null {
  const cached = cache.get(key);
  if (cached && Date.now() - cached.timestamp < maxAge) {
    return cached.data;
  }
  return null;
}

function setCachedData(key: string, data: any): void {
  cache.set(key, { data, timestamp: Date.now() });
}

// Utility function to make API requests with timeout
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = REQUEST_TIMEOUT): Promise {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  if (options.signal) {
    options.signal.addEventListener('abort', () => controller.abort());
  }

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    
    // Global API Error Interceptor
    if (!response.ok) {
      console.error(
        `[API Error] ${new Date().toISOString()} | HTTP ${response.status} ${response.statusText} | URL: ${url}`
      );
    }
    
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      if (options.signal?.aborted) {
        throw error;
      }
      throw new Error('Request timed out. Please try again.');
    }
    throw error;
  }
}

let activeFetchControllers: Record<string, AbortController> = {};
const inflightLivePriceRequests = new Map<string, Promise<LivePriceResponse>>();

export const stockService = {
  cancelActiveRequests: (requestKey?: string) => {
    if (requestKey) {
      if (activeFetchControllers[requestKey]) {
        activeFetchControllers[requestKey].abort();
        delete activeFetchControllers[requestKey];
      }
    } else {
      Object.values(activeFetchControllers).forEach(controller => controller.abort());
      activeFetchControllers = {};
    }
  },
  
  getNewAbortSignal: (requestKey: string = 'default') => {
    stockService.cancelActiveRequests(requestKey);
    activeFetchControllers[requestKey] = new AbortController();
    return activeFetchControllers[requestKey].signal;
  },

  // Get stock metadata quickly (without live price)
  getStockInfo: async (symbol: string): Promise => {
    if (!symbol || !symbol.trim()) {
      throw new Error('Stock symbol is required');
    }

    const cleanSymbol = symbol.trim().toUpperCase();
    const cacheKey = `stock_info_${cleanSymbol}`;
    
    // Check cache first (5 minutes cache)
    const cachedData = getCachedData(cacheKey, 5 * 60 * 1000);
    if (cachedData) {
      return cachedData;
    }

    try {
      const signal = stockService.getNewAbortSignal('stockInfo');
      const response = await fetchWithTimeout(`${BACKEND_BASE_URL}/stock_info?symbol=${encodeURIComponent(cleanSymbol)}`, { signal });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result: ApiResponse = await response.json();

      if (!result.success || !result.data) {
        throw new Error(result.message || 'Failed to fetch stock info');
      }

      setCachedData(cacheKey, result.data);
      return result.data;

    } catch (error) {
      console.error(`Failed to fetch stock info for ${cleanSymbol}:`, error);
      if (error instanceof Error) {
        if (error.message.includes('timed out')) {
          throw new Error('Request timed out. Please try again.');
        } else if (error.message.includes('Failed to fetch')) {
          throw new Error('Unable to connect to server. Please ensure the backend is running.');
        } else {
          throw error;
        }
      }
      throw new Error(`Unable to fetch stock info for ${cleanSymbol}. Please try again later.`);
    }
  },

  // Get live stock price from backend
  getLivePrice: async (symbol: string, forceRefresh: boolean = false): Promise => {
    if (!symbol || !symbol.trim()) {
      throw new Error('Stock symbol is required');
    }

    const cleanSymbol = symbol.trim().toUpperCase();
    const cacheKey = `live_price_${cleanSymbol}`;
    
    // Only use cache if not forcing refresh
    if (!forceRefresh) {
      const cachedData = getCachedData(cacheKey, 2 * 60 * 1000); // 2 minutes cache
      if (cachedData) {
        return cachedData;
      }
      
      if (inflightLivePriceRequests.has(cleanSymbol)) {
        return inflightLivePriceRequests.get(cleanSymbol)!;
      }
    }

    const fetchPromise = (async () => {
      try {
        const signal = stockService.getNewAbortSignal(`livePrice_${cleanSymbol}`);
      const response = await fetchWithTimeout(`${BACKEND_BASE_URL}/live_price?symbol=${encodeURIComponent(cleanSymbol)}`, { signal });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result: ApiResponse = await response.json();

      if (!result.success || !result.data) {
        throw new Error(result.message || 'Failed to fetch live price');
      }

      setCachedData(cacheKey, result.data);
      return result.data;

    } catch (error) {
      console.error(`Failed to fetch live price for ${cleanSymbol}:`, error);
      if (error instanceof Error) {
        if (error.message.includes('timed out')) {
          throw new Error('Request timed out. Please try again.');
        } else if (error.message.includes('Failed to fetch')) {
          throw new Error('Unable to connect to server. Please ensure the backend is running.');
        } else {
          throw error;
        }
      }
      throw new Error(`Unable to fetch live price for ${cleanSymbol}. Please try again later.`);
    } finally {
      inflightLivePriceRequests.delete(cleanSymbol);
    }
    })();
    
    inflightLivePriceRequests.set(cleanSymbol, fetchPromise);
    return fetchPromise;
  },

  // Get current stock data (converted from live price for compatibility)
  getStockData: async (symbol: string): Promise => {
    if (!symbol || !symbol.trim()) {
      throw new Error('Stock symbol is required');
    }

    const cleanSymbol = symbol.trim().toUpperCase();

    try {
      const livePrice = await stockService.getLivePrice(cleanSymbol);

      // IMPORTANT: Convert all prices to USD FIRST (models are trained on USD-normalized data)
      const priceInUSD = livePrice.currency === 'INR' ? (livePrice.price_usd || livePrice.price) : livePrice.price;
      const openInUSD = livePrice.currency === 'INR' && livePrice.open ? (livePrice.open / (livePrice.exchange_rate || 1)) : livePrice.open;
      const highInUSD = livePrice.currency === 'INR' && livePrice.high ? (livePrice.high / (livePrice.exchange_rate || 1)) : livePrice.high;
      const lowInUSD = livePrice.currency === 'INR' && livePrice.low ? (livePrice.low / (livePrice.exchange_rate || 1)) : livePrice.low;

      // Calculate change and change percent AFTER USD conversion
      let change = 0;
      let changePercent = 0;
      
      if (openInUSD && openInUSD > 0) {
        change = priceInUSD - openInUSD;
        changePercent = (change / openInUSD) * 100;
      }
      
      const stockData: StockData = {
        symbol: livePrice.symbol || cleanSymbol,
        name: livePrice.company_name,
        price: priceInUSD,  // Always in USD for model compatibility
        change: change,
        changePercent: changePercent,
        volume: livePrice.volume || 0,
        marketCap: livePrice.market_cap || 'N/A',
        lastUpdated: livePrice.timestamp,
        open: openInUSD,
        high: highInUSD,
        low: lowInUSD
      };

      return stockData;
    } catch (error) {
      console.error(`Failed to get stock data for ${cleanSymbol}:`, error);
      throw error;
    }
  },

  // Get historical data from backend
  getHistoricalData: async (symbol: string, period: 'year' | '5year'): Promise => {
    if (!symbol || !symbol.trim()) {
      throw new Error('Stock symbol is required');
    }

    const cleanSymbol = symbol.trim().toUpperCase();
    const cacheKey = `historical_${cleanSymbol}_${period}`;
    
    // Check cache first (1 hour cache)
    const cachedData = getCachedData(cacheKey, 60 * 60 * 1000);
    if (cachedData) {
      return cachedData;
    }

    try {
      // Fetch historical data from backend
      const response = await fetchWithTimeout(
        `${BACKEND_BASE_URL}/historical?symbol=${encodeURIComponent(cleanSymbol)}&period=${period}`
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result: ApiResponse = await response.json();

      if (!result.success || !result.data) {
        throw new Error(result.message || 'Failed to fetch historical data');
      }

      // Map backend response to PricePoint format
      const historicalData: PricePoint[] = result.data.map(point => ({
        date: point.date,
        open: point.open,
        high: point.high,
        low: point.low,
        close: point.close,
        volume: point.volume,
        price: point.close, // Map close to price for chart display
        currency: point.currency || 'USD' // Include currency from backend
      }));

      // If no historical data found and period is 5year, try fallback to 1 year
      if (historicalData.length === 0 && period === '5year') {
        console.warn(`No 5-year data found for ${cleanSymbol}, trying 1-year fallback`);
        return await stockService.getHistoricalData(cleanSymbol, 'year');
      }

      // Append latest live price as the most recent data point
      try {
        const livePrice = await stockService.getLivePrice(cleanSymbol);
        const today = new Date().toISOString().split('T')[0];
        
        // Check if we already have today's data
        const hasTodayData = historicalData.some(point => point.date === today);
        
        if (!hasTodayData) {
          const livePricePoint: PricePoint = {
            date: today,
            open: livePrice.price,
            high: livePrice.price,
            low: livePrice.price,
            close: livePrice.price,
            volume: 0,
            price: livePrice.price,
            currency: livePrice.currency || 'USD'
          };
          historicalData.push(livePricePoint);
        }
      } catch (livePriceError) {
        console.warn(`Could not fetch live price for ${cleanSymbol}:`, livePriceError);
        // Continue without live price data
      }

      setCachedData(cacheKey, historicalData);
      return historicalData;

    } catch (error) {
      console.error(`Failed to fetch historical data for ${cleanSymbol}:`, error);
      if (error instanceof Error) {
        if (error.message.includes('timed out')) {
          throw new Error('Request timed out. Please try again.');
        } else if (error.message.includes('Failed to fetch')) {
          throw new Error('Unable to connect to server. Please ensure the backend is running.');
        } else {
          throw error;
        }
      }
      throw new Error(`Unable to fetch historical data for ${cleanSymbol}. Please try again later.`);
    }
  },

  // Search stocks with backend integration
  searchStocks: async (query: string): Promise<{ symbol: string; name: string }[]> => {
    const cleanQuery = query ? query.trim() : '';
    console.log(`🔍 Searching for: "${cleanQuery}"`);

    // Popular stocks for fallback
    const popularStocks = [
      { symbol: 'AAPL', name: 'Apple Inc.' },
      { symbol: 'GOOGL', name: 'Alphabet Inc.' },
      { symbol: 'MSFT', name: 'Microsoft Corporation' },
      { symbol: 'AMZN', name: 'Amazon.com Inc.' },
      { symbol: 'TSLA', name: 'Tesla Inc.' },
      { symbol: 'META', name: 'Meta Platforms Inc.' },
      { symbol: 'NVDA', name: 'NVIDIA Corporation' },
      { symbol: 'NFLX', name: 'Netflix Inc.' }
    ];

    if (!cleanQuery) {
      console.log(`📋 Returning popular stocks for empty query`);
      return popularStocks;
    }
    
    const cacheKey = `search_${cleanQuery.toLowerCase()}`;
    const cachedData = getCachedData<{ symbol: string; name: string }[]>(cacheKey, 5 * 60 * 1000); // 5 minutes cache

    if (cachedData) {
      console.log(`📦 Using cached data for: "${cleanQuery}"`);
      return cachedData;
    }

    try {
      const url = `${BACKEND_BASE_URL}/search?q=${encodeURIComponent(cleanQuery)}`;
      console.log(`🌐 Making request to: ${url}`);
      
      const response = await fetchWithTimeout(url);
      console.log(`📡 Response status: ${response.status}`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result: ApiResponse<{ symbol: string; name: string }[]> = await response.json();
      console.log(`📊 Backend response:`, result);

      if (result.success && result.data && result.data.length > 0) {
        console.log(`✅ Found ${result.data.length} results from backend`);
        setCachedData(cacheKey, result.data);
        return result.data;
      } else {
        console.log(`⚠️ Backend returned no results, using fallback`);
        // Fallback to hardcoded popular stocks if backend returns no results
        const filteredPopular = popularStocks.filter(stock =>
          stock.symbol.toLowerCase().includes(cleanQuery.toLowerCase()) ||
          stock.name.toLowerCase().includes(cleanQuery.toLowerCase())
        );
        return filteredPopular;
      }

    } catch (error) {
      console.error(`❌ Failed to search stocks for "${cleanQuery}":`, error);

      // Fallback to hardcoded popular stocks on error
      const filteredPopular = popularStocks.filter(stock =>
        stock.symbol.toLowerCase().includes(cleanQuery.toLowerCase()) ||
        stock.name.toLowerCase().includes(cleanQuery.toLowerCase())
      );
      console.log(`🔄 Using fallback with ${filteredPopular.length} results`);
      return filteredPopular;
    }
  },

  // Get popular stocks
  getPopularStocks: async (): Promise<{ symbol: string; name: string }[]> => {
    return [
      { symbol: 'AAPL', name: 'Apple Inc.' },
      { symbol: 'GOOGL', name: 'Alphabet Inc.' },
      { symbol: 'MSFT', name: 'Microsoft Corporation' },
      { symbol: 'AMZN', name: 'Amazon.com Inc.' },
      { symbol: 'TSLA', name: 'Tesla Inc.' },
      { symbol: 'META', name: 'Meta Platforms Inc.' },
      { symbol: 'NVDA', name: 'NVIDIA Corporation' },
      { symbol: 'NFLX', name: 'Netflix Inc.' }
    ];
  },

  // Health check for backend status
  checkHealth: async (): Promise<{ status: string; timestamp: string }> => {
    try {
      const response = await fetchWithTimeout(`${BACKEND_BASE_URL}/health`);

      if (!response.ok) {
        throw new Error(`Backend health check failed: ${response.status}`);
      }

      const result: ApiResponse<{ status: string; timestamp: string }> = await response.json();

      if (!result.success || !result.data) {
        throw new Error('Backend health check failed');
      }

      return result.data;

    } catch (error) {
      console.error('Backend health check failed:', error);
      throw new Error('Backend server is unavailable');
    }
  },

  // Get ML prediction for a stock
  getPrediction: async (symbol: string, horizon: string = '1d', model?: string, currentPrice?: number): Promise => {
    const cleanSymbol = symbol ? symbol.trim().toUpperCase() : '';
    try {
      const params = new URLSearchParams({
        symbol: cleanSymbol,
        horizon
      });
      
      if (model && model !== 'all') {
        params.append('model', model);
      }
      
      // Pass live current price to backend to avoid duplicate fetching
      if (currentPrice) {
        params.append('current_price', currentPrice.toString());
      }

      const url = `${BACKEND_BASE_URL}/api/predict?${params.toString()}`;
      console.log(`🌐 Making prediction request to: ${url}`);
      
      const response = await fetchWithTimeout(url, {}, 30000); // 30 second timeout for ML predictions
      console.log(`📡 Prediction response status: ${response.status}`);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      console.log(`📊 Prediction result:`, result);

      // Convert backend response to frontend format
      const prediction: PredictionResult = {
        predictedPrice: result.predicted_price,
        currentPrice: result.current_price,  // Current price in USD (same currency as predicted_price)
        confidence: result.confidence,
        model: result.model_info?.model || 'Ensemble',
        timeframe: horizon,
        priceRange: result.price_range,
        timeFrameDays: result.time_frame_days,
        modelInfo: result.model_info,
        dataPointsUsed: result.data_points_used,
        lastUpdated: result.last_updated,
        currency: result.currency,
        dataSource: result.data_source,  // 'live_api' or 'stored_data'
        dataDate: result.data_date,  // Date of stored data (null for live)
        sourceReliable: result.data_source === 'live_api'  // true if live, false if stored
      };

      return prediction;

    } catch (error) {
      console.error(`❌ Failed to get prediction for ${cleanSymbol}:`, error);
      throw new Error(`Prediction failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
};
