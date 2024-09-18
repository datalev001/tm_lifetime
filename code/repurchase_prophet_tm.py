# Import necessary libraries
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import warnings
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')


# Load the dataset with proper encoding
tran_df = pd.read_csv('online_retail_II.csv', encoding= "latin1")

# This step filters out rows that contain missing or invalid values in the key columns.
c1 = (tran_df['Invoice'].isnull() == False)
c2 = (tran_df['Quantity']>0)
c3 = (tran_df['Customer ID'].isnull() == False)
c4 = (tran_df['StockCode'].isnull() == False)
c5 = (tran_df['Description'].isnull() == False)
tran_df = tran_df[c1 & c2 & c3 & c4 & c5]

# This step involves further cleaning and filtering
grp = ['Invoice', 'StockCode','Description', 'Quantity', 'InvoiceDate']
# Duplicate transactions are removed
tran_df = tran_df.drop_duplicates(grp)
# Converted into a standardized datetime format
tran_df['InvoiceDate'] = pd.to_datetime(tran_df['InvoiceDate'])
tran_df['transaction_date'] = tran_df['InvoiceDate'].dt.date

cats_top = tran_df.Description.value_counts().reset_index()
cats_top_df = cats_top[cats_top['count']>1000]

# Filtered to keep only the high-frequency items:
pro_lst = list(set(cats_top_df['Description']))
tran_df_sel = tran_df[tran_df['Description'].isin(pro_lst)]
cols = ['Customer ID', 'Description', 'trans_date', 'Quantity']

## data to be used
tran_df_bs = tran_df_sel[cols]

######repurchase model####################

# Assuming 'tran_df_bs' is your original DataFrame
df = tran_df_bs.copy()

# Convert 'trans_date' to datetime
df['trans_date'] = pd.to_datetime(df['trans_date'])

# Handle missing values and data cleaning
df.dropna(subset=['Customer ID', 'Quantity'], inplace=True)
df = df[df['Quantity'] > 0]  # Remove negative quantities

# Remove outliers in 'Quantity' using IQR method
Q1 = df['Quantity'].quantile(0.25)
Q3 = df['Quantity'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Quantity'] >= Q1 - 1.5 * IQR) & (df['Quantity'] <= Q3 + 1.5 * IQR)]

# Split the data into training and validation sets
cutoff_date = pd.to_datetime('2011-11-30')
train_df = df[df['trans_date'] < cutoff_date]
valid_df = df[(df['trans_date'] >= cutoff_date) & (df['trans_date'] < cutoff_date + pd.Timedelta(days=10))]

# Update valid_df to include only customers present in train_df
train_customers = train_df['Customer ID'].unique()
valid_df = valid_df[valid_df['Customer ID'].isin(train_customers)]

# Prepare data for the Lifetimes model
summary = summary_data_from_transaction_data(
    train_df,
    'Customer ID',
    'trans_date',
    monetary_value_col='Quantity',
    observation_period_end=train_df['trans_date'].max()
)

# Ensure monetary values are positive
summary = summary[summary['monetary_value'] > 0]

# Fit the BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.05)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])

# Predict the number of transactions for the next 10 days
t = 10  # prediction period matches validation period
summary['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    t, summary['frequency'], summary['recency'], summary['T']
)
summary['predicted_purchases'] = summary['predicted_purchases'].clip(lower=0)

# Fit the Gamma-Gamma model for monetary value
returning_customers_summary = summary[summary['frequency'] > 0]
ggf = GammaGammaFitter(penalizer_coef=0.02)
ggf.fit(returning_customers_summary['frequency'], returning_customers_summary['monetary_value'])
summary['expected_avg_sales'] = ggf.conditional_expected_average_profit(
    summary['frequency'],
    summary['monetary_value']
)
summary['expected_avg_sales'] = summary['expected_avg_sales'].clip(lower=0)

# Calculate expected sales for each customer
summary['expected_sales'] = summary['predicted_purchases'] * summary['expected_avg_sales']
summary['expected_sales'] = summary['expected_sales'].clip(lower=0)

# Merge predictions with customer-product data
customer_product = train_df.groupby(['Customer ID', 'Description'])['Quantity'].sum().reset_index()
customer_product = customer_product.merge(
    summary[['expected_sales', 'frequency', 'recency', 'monetary_value']],
    left_on='Customer ID', right_index=True, how='left'
)

# Calculate the proportion of each product purchased by each customer
total_quantity_per_customer = customer_product.groupby('Customer ID')['Quantity'].transform('sum')
customer_product['product_proportion'] = customer_product['Quantity'] / total_quantity_per_customer
customer_product['product_proportion'] = customer_product['product_proportion'].clip(lower=0)

# Calculate expected sales per product
customer_product['expected_product_sales'] = customer_product['expected_sales'] * customer_product['product_proportion']
customer_product['expected_product_sales'] = customer_product['expected_product_sales'].clip(lower=0)

# Aggregate expected sales by product
product_sales_forecast = customer_product.groupby('Description')['expected_product_sales'].sum().reset_index()

# --- Improved Seasonal Adjustment ---

# Aggregate data to daily sales per product
daily_sales = train_df.groupby(['trans_date', 'Description'])['Quantity'].sum().reset_index()

# Pivot data to have products as columns
daily_sales_pivot = daily_sales.pivot(index='trans_date', columns='Description', values='Quantity').fillna(0)

# Decompose each product's time series
seasonal_indices = {}
for product in daily_sales_pivot.columns:
    product_series = daily_sales_pivot[product]
    has_zeros = (product_series == 0).any()
    model_type = 'additive' if has_zeros else 'multiplicative'
    if len(product_series.dropna()) >= 90:
        try:
            decomposition = seasonal_decompose(product_series, model=model_type, period=30)
            seasonal = decomposition.seasonal
            forecast_dates = pd.date_range(start=cutoff_date, periods=t)
            forecast_seasonal = seasonal[seasonal.index.isin(forecast_dates)]
            if not forecast_seasonal.empty:
                seasonal_index = forecast_seasonal.mean()
                seasonal_indices[product] = {'index': seasonal_index, 'model': model_type}
            else:
                seasonal_indices[product] = {'index': 0 if model_type == 'additive' else 1, 'model': model_type}
        except Exception as e:
            print(f"Seasonal decomposition failed for product '{product}': {e}")
            seasonal_indices[product] = {'index': 0 if model_type == 'additive' else 1, 'model': model_type}
    else:
        seasonal_indices[product] = {'index': 0 if model_type == 'additive' else 1, 'model': model_type}

# Apply seasonal adjustment
def adjust_sales(row):
    product = row['Description']
    expected_sales = row['expected_product_sales']
    seasonal_info = seasonal_indices.get(product, {'index': 0, 'model': 'additive'})
    seasonal_index = seasonal_info['index']
    model_type = seasonal_info['model']
    if model_type == 'additive':
        adjusted_sales = expected_sales + seasonal_index
    else:
        adjusted_sales = expected_sales * seasonal_index
    return max(adjusted_sales, 0)

product_sales_forecast['adjusted_expected_sales'] = product_sales_forecast.apply(adjust_sales, axis=1)


# Final forecasted total sales for each product
final_forecast = product_sales_forecast[['Description', 'adjusted_expected_sales']]

# Validate the forecast
actual_sales = valid_df.groupby('Description')['Quantity'].sum().reset_index()
actual_sales.rename(columns={'Quantity': 'actual_sales'}, inplace=True)

# Merge with forecasted sales
validation_df = final_forecast.merge(actual_sales, on='Description', how='left')
validation_df['actual_sales'].fillna(0, inplace=True)

# Calculate APE
validation_df['APE'] = np.where(
    validation_df['actual_sales'] == 0,
    np.nan,
    np.abs((validation_df['actual_sales'] - validation_df['adjusted_expected_sales']) / validation_df['actual_sales'])
)

# Calculate MAE and RMSE
validation_df['Error'] = validation_df['adjusted_expected_sales'] - validation_df['actual_sales']
mae = mean_absolute_error(validation_df['actual_sales'], validation_df['adjusted_expected_sales'])
rmse = np.sqrt(mean_squared_error(validation_df['actual_sales'], validation_df['adjusted_expected_sales']))

print(f"Validation MAE: {mae:.2f}")
print(f"Validation RMSE: {rmse:.2f}")

# Filter out low sales values
threshold = 5
validation_df_filtered = validation_df[validation_df['actual_sales'] >= threshold]

# Calculate MAPE and median APE
mape = validation_df_filtered['APE'].mean(skipna=True) * 100
median_ape = validation_df_filtered['APE'].median(skipna=True) * 100
validation_df_filtered.shape

print(f"Filtered Validation MAPE: {mape:.2f}%")
print(f"Filtered Median APE: {median_ape:.2f}%")

# Display the final forecast
print("\nFinal Sales Forecast for Each Product in the Next 10 Days:")
print(validation_df_filtered[['Description', 'adjusted_expected_sales', 'actual_sales', 'APE']])



########prophet model#########

df = remove_outliers_iqr(df)

# Aggregate data to daily sales per product
product_daily_sales = df.groupby(['Description', 'trans_date'])['Quantity'].sum().reset_index()

# Split the data into training and validation sets
cutoff_date = pd.to_datetime('2011-11-05')
train_data = product_daily_sales[product_daily_sales['trans_date'] < cutoff_date]
valid_data = product_daily_sales[(product_daily_sales['trans_date'] >= cutoff_date) & (product_daily_sales['trans_date'] < cutoff_date + pd.Timedelta(days=30))]

# Get list of unique products
products = df['Description'].unique()

# Initialize a DataFrame to store forecasts
forecast_results = pd.DataFrame()

# Loop over each product and build a Prophet model
for product in products:
    product_train = train_data[train_data['Description'] == product][['trans_date', 'Quantity']]
    product_valid = valid_data[valid_data['Description'] == product][['trans_date', 'Quantity']]
    
    # Check if there is enough data to train the model
    if len(product_train) < 2:
        continue  # Skip products with insufficient data
    
    # Prepare the data for Prophet
    product_train = product_train.rename(columns={'trans_date': 'ds', 'Quantity': 'y'})
    product_valid = product_valid.rename(columns={'trans_date': 'ds', 'Quantity': 'y'})
    
    # Instantiate and fit the Prophet model, adding monthly seasonality
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    # Add monthly seasonality to the model
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    try:
        model.fit(product_train)
    except Exception as e:
        print(f"Model fitting failed for product '{product}': {e}")
        continue  # Skip this product if the model fails to fit
    
    # Create a DataFrame to hold future dates
    future_dates = model.make_future_dataframe(periods=30, freq='D')
    
    # Generate the forecast
    forecast = model.predict(future_dates)
    
    # Extract the forecasted values for the next 30 days
    forecast_next_30 = forecast[['ds', 'yhat']].tail(30)
    forecast_next_30['Description'] = product
    
    # Merge with actual validation data
    validation = product_valid.merge(forecast_next_30, on='ds', how='left')
    validation['yhat'].fillna(0, inplace=True)
    validation['y'].fillna(0, inplace=True)
    
    # Calculate Absolute Percentage Error
    validation['APE'] = np.where(
        validation['y'] == 0,
        np.nan,
        np.abs((validation['y'] - validation['yhat']) / validation['y'])
    )
    
    # Append to the forecast results
    forecast_results = pd.concat([forecast_results, validation[['Description', 'ds', 'y', 'yhat', 'APE']]], ignore_index=True)

# Calculate overall MAPE
mape = forecast_results['APE'].mean(skipna=True) * 100
median_ape = forecast_results['APE'].median(skipna=True) * 100

print(f"Validation MAPE: {mape:.2f}%")
print(f"Median APE: {median_ape:.2f}%")

# Aggregate forecasts to get total expected sales per product
product_forecasts = forecast_results.groupby('Description')['yhat'].sum().reset_index()
product_forecasts.rename(columns={'yhat': 'forecasted_sales'}, inplace=True)

# Merge with actual sales
actual_sales = valid_data.groupby('Description')['Quantity'].sum().reset_index()
actual_sales.rename(columns={'Quantity': 'actual_sales'}, inplace=True)

validation_df = product_forecasts.merge(actual_sales, on='Description', how='left')
validation_df['actual_sales'].fillna(0, inplace=True)

# Calculate APE per product
validation_df['APE'] = np.where(
    validation_df['actual_sales'] == 0,
    np.nan,
    np.abs((validation_df['actual_sales'] - validation_df['forecasted_sales']) / validation_df['actual_sales'])
)

# Filter out low sales values
threshold = 5
validation_df_filtered = validation_df[validation_df['actual_sales'] >= threshold]

mape = validation_df_filtered['APE'].mean(skipna=True) * 100
median_ape = validation_df_filtered['APE'].median(skipna=True) * 100
validation_df_filtered.shape

print(f"Filtered Validation MAPE: {mape:.2f}%")
print(f"Filtered Median APE: {median_ape:.2f}%")

# Display the final forecast with validation metrics
print("\nFinal Sales Forecast for Each Product in the Next 30 Days:")
print(validation_df[['Description', 'forecasted_sales', 'actual_sales', 'APE']])
