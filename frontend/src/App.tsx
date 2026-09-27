import React, { useState, useEffect } from 'react';
import { BarChart3, AlertCircle, Info } from 'lucide-react';
import { StockSearch } from './components/StockSearch';
import { StockInfo } from './components/StockInfo';
import { StockChart } from './components/StockChart';
import { StockPrediction } from './components/StockPrediction';
import { CurrencyToggle } from './components/CurrencyToggle';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card';
import { Alert, AlertDescription } from './components/ui/alert';
import { stockService, StockData, PricePoint, PredictionResult, LivePriceResponse, StockInfoResponse } from './services/stockService';
import { Currency } from './utils/currency';
import { Toaster } from './components/ui/sonner';
import { toast } from 'sonner';

export default function App() {
  const [selectedSymbol, setSelectedSymbol] = useState<string>('');
  const [stockData, setStockData] = useState<StockData | null>(null);
  const [livePriceData, setLivePriceData] = useState<LivePriceResponse | null>(null);
  const [stockInfoData, setStockInfoData] = useState<StockInfoResponse | null>(null);
  const [chartData, setChartData] = useState<PricePoint[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predictionHorizon, setPredictionHorizon] = useState<string>('1D');
  const [selectedModel, setSelectedModel] = useState<string>('linear_regression');
  const [chartPeriod, setChartPeriod] = useState<'year' | '5year'>('year');
  const [currency, setCurrency] = useState<Currency>('USD');
  const [loading, setLoading] = useState({
    stock: false,
    chart: false,
    prediction: false
  });
  const [errors, setErrors] = useState({
    stock: '',
    chart: '',
    prediction: ''
  });

  // Load stock data when symbol changes
  useEffect(() => {
    if (selectedSymbol) {
      loadStockDataAndPrediction(selectedSymbol);
      loadChartData(selectedSymbol, chartPeriod);
    }
  }, [selectedSymbol]);

  // Reload prediction when horizon changes
  useEffect(() => {
    if (selectedSymbol && stockData?.price) {
      loadPrediction(selectedSymbol, predictionHorizon, selectedModel, stockData.price);
    }
  }, [predictionHorizon, selectedSymbol]);

  // Reload prediction when model changes
  useEffect(() => {
    if (selectedSymbol && stockData?.price) {
      loadPrediction(selectedSymbol, predictionHorizon, selectedModel, stockData.price);
    }
  }, [selectedModel, selectedSymbol, predictionHorizon]);

  // Reload chart data when period changes
  useEffect(() => {
    if (selectedSymbol) {
      loadChartData(selectedSymbol, chartPeriod);
    }
  }, [chartPeriod, selectedSymbol]);



  const loadChartData = async (symbol: string, period: 'year' | '5year') => {
    setLoading(prev => ({ ...prev, chart: true }));
    setErrors(prev => ({ ...prev, chart: '' }));
    try {
      const data = await stockService.getHistoricalData(symbol, period);
      setChartData(data);
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Chart data load aborted');
        return;
      }
      console.error('Failed to load chart data:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to load chart data';
      toast.error('Chart Data Error', { description: errorMessage });
      setChartData([]);
    } finally {
      setLoading(prev => ({ ...prev, chart: false }));
    }
  };

  const loadPrediction = async (
    symbol: string, 
    horizon: string = predictionHorizon,
    model: string = selectedModel,
    currentPrice?: number,
    dataSource?: string,
    sourceReliable?: boolean
  ) => {
    setLoading(prev => ({ ...prev, prediction: true }));
    setErrors(prev => ({ ...prev, prediction: '' }));
    try {
      const data = await stockService.getPrediction(symbol, horizon, model, currentPrice);
      // Add source information if provided
      if (dataSource !== undefined) {
        data.dataSource = dataSource;
        data.sourceReliable = sourceReliable;
      }
      setPrediction(data);
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Prediction load aborted');
        return;
      }
      console.error('Failed to load prediction:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate prediction';
      toast.error('Prediction Error', { description: errorMessage });
      setPrediction(null);
    } finally {
      setLoading(prev => ({ ...prev, prediction: false }));
    }
  };

  const loadStockDataAndPrediction = async (symbol: string, forceRefresh: boolean = false) => {
    // Step 1: Load stock data (gets live price)
    setLoading(prev => ({ ...prev, stock: true }));
    setErrors(prev => ({ ...prev, stock: '' }));
    
    try {
      // Fetch stock info quickly (metadata)
      try {
        const stockInfo = await stockService.getStockInfo(symbol);
        setStockInfoData(stockInfo);
      } catch (error) {
        console.warn('Failed to load stock info:', error);
      }

      // Fetch live price data
      const livePrice = await stockService.getLivePrice(symbol, forceRefresh);
      setLivePriceData(livePrice);

      // Convert to StockData format
      const data = await stockService.getStockData(symbol);
      setStockData(data);
      
      // Step 2: Now that we have the live price, load prediction with it
      // KEEP EXISTING FLOW: Pass current_price to avoid duplicate fetch
      if (data?.price && livePrice) {
        // Use the price we already fetched - NO duplicate API call
        // Pass source info so both cards show the same data source
        await loadPrediction(symbol, predictionHorizon, selectedModel, data.price, livePrice.source, livePrice.source_reliable);
      } else {
        // If live price failed, don't fall back to stale data - show error
        setPrediction(null);
        toast.error('Prediction Error', { description: 'No live price available for prediction' });
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Stock data load aborted');
        return;
      }
      console.error('Failed to load stock data:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to load stock data';
      toast.error('Stock Data Error', { description: errorMessage });
      setStockData(null);
      setLivePriceData(null);
      setStockInfoData(null);
      setPrediction(null);
    } finally {
      setLoading(prev => ({ ...prev, stock: false }));
    }
  };

  const handleStockSelect = (symbol: string) => {
    stockService.cancelActiveRequests(); // Cancel any inflight requests!
    setSelectedSymbol(symbol);
    setStockData(null);
    setLivePriceData(null);
    setStockInfoData(null);
    setPrediction(null);
  };

  const handlePeriodChange = (period: 'year' | '5year') => {
    setChartPeriod(period);
  };

  const handleCurrencyChange = (newCurrency: Currency) => {
    setCurrency(newCurrency);
  };

  const handleHorizonChange = (horizon: string) => {
    setPredictionHorizon(horizon);
  };

  const handleModelChange = (model: string) => {
    setSelectedModel(model);
  };

  return (
    <div className="min-h-screen bg-background">
      <Toaster position="top-right" richColors />
      <div className="main-app-container space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="header-container">
            <BarChart3 className="stock-price-prediction-icon text-primary" />
            <h1 className="stock-price-prediction-header">Stock Price Prediction</h1>
          </div>
          <p className="text-2xl text-muted-foreground max-w-2xl mx-auto">
            Analyze real-time stock data and get AI-powered price predictions using machine learning models.
            Always conduct your own research before making investment decisions.
          </p>
        </div>

        {/* Warning Banner */}
        <Alert className="alert-scaled">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Investment Warning:</strong> Stock market predictions are inherently uncertain.
            This tool provides statistical analysis for educational purposes only. Past performance
            does not guarantee future results. Always consult with financial advisors and do thorough
            research before making investment decisions.
          </AlertDescription>
        </Alert>

        <div className="card-grid-layout">
          {/* Left Sidebar */}
          <div className="sidebar-container">
            <StockSearch
              onStockSelect={handleStockSelect}
              selectedSymbol={selectedSymbol}
            />
            <StockInfo
              data={stockData}
              loading={loading.stock}
              error={errors.stock}
              currency={currency}
              onCurrencyChange={handleCurrencyChange}
              livePriceData={livePriceData}
              stockInfoData={stockInfoData}
              onRefresh={() => selectedSymbol && loadStockDataAndPrediction(selectedSymbol, true)}
            />
          </div>

          {/* Main Content */}
          <div className="main-content-container">
            <StockChart
              data={chartData}
              symbol={selectedSymbol || 'Select a Stock'}
              onPeriodChange={handlePeriodChange}
              currentPeriod={chartPeriod}
              loading={loading.chart}
              error={errors.chart}
              currency={currency}
            />

            <StockPrediction
              prediction={prediction}
              currentPrice={stockData?.price}
              loading={loading.prediction}
              symbol={selectedSymbol}
              error={errors.prediction}
              currency={currency}
              selectedHorizon={predictionHorizon}
              selectedModel={selectedModel}
              onHorizonChange={handleHorizonChange}
              onModelChange={handleModelChange}
              dataTimestamp={livePriceData?.timestamp}
              dataSource={livePriceData?.source}
            />
          </div>
        </div>

        {/* Footer */}
        <Card>
          <CardHeader>
            <CardTitle className="card-title-scaled card-title-with-icon">
              <Info className="card-icon-scaled" />
              About This Tool
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-2xl text-muted-foreground">
              This stock analysis dashboard provides real-time stock data, historical charts, and experimental 
              machine learning predictions. The system uses multiple ML algorithms including Random Forest, 
              Linear Regression, and Neural Networks to analyze price patterns and generate predictions.
            </p>
            <div className="text-2xl text-muted-foreground space-y-2">
              <p><strong>Backend:</strong> Powered by Flask with real-time data processing and ML model integration</p>
              <p><strong>Prediction Models:</strong> 7 ML algorithms including Random Forest, Linear Regression, SVM, Decision Tree, KNN, ARIMA, and Autoencoder</p>
              <p><strong>Model Categories:</strong> 4 Basic Models (supervised learning) + 3 Advanced Models (time series & deep learning)</p>
              <p><strong>Timeframes:</strong> Supports 1D, 1W, 1M, 1Y, and 5Y prediction horizons</p>
              <p><strong>Historical Data:</strong> 5-year dataset (2020-2025) with daily granularity for both US and Indian stocks</p>
              <p><strong>Currency Support:</strong> Real-time conversion between USD and INR with live formatting</p>
              <p><strong>Status:</strong> ML prediction system fully operational - All models trained on percentage change data</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}