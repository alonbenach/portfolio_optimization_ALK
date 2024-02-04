# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.ticker import FuncFormatter, MaxNLocator
import seaborn as sns
from scipy.optimize import minimize
import sklearn as sklearn
from scipy.optimize import minimize
import gc
import statsmodels.api as sm

# %%
# Load the data
set_A = pd.read_csv("data/set_A.csv")
set_B = pd.read_csv("data/set_B.csv")

# %%
# Examine set_A
print(
    f"""Set A - head: \n {set_A.head()}
Set A - description: \n {set_A.describe()}
"""
)
# Check which columns have missing values in set_A
col_with_missing_A = set_A.columns[set_A.isna().any()].tolist()
print(f"Columns with NAs: \n {col_with_missing_A}")


# %%
# Examine set_B
print(
    f"""Set B - head: \n {set_B.head()}
Set B - description: \n {set_B.describe()}
"""
)
# Check which columns have missing values in set_B
col_with_missing_B = set_B.columns[set_B.isna().any()].tolist()
print(f"Columns with NAs: \n {col_with_missing_B}")

# %% [markdown]
# * Since in set_A iddate3 contains NA rows, nazwapl will be cleaned to obtain a column of ticker names from both sets.
# * The dataframe obtained from concatenating the sets will only subset the ticker names, closing price and date.

# %%
# Extract ticker from the 'nazwapl' column
set_A["Ticker"] = set_A["nazwapl"].str.extract("\\[\\'(.*?)\\'\\]")
set_B["Ticker"] = set_B["nazwapl"].str.extract("\\[\\'(.*?)\\'\\]")


# Display the updated sets with the new 'Ticker' column
print(set_A[["nazwapl", "Ticker"]].head())
print(set_A[["nazwapl", "Ticker"]].tail())
print(set_B[["nazwapl", "Ticker"]].head())
print(set_B[["nazwapl", "Ticker"]].tail())

# %%
# Convert the Date column to a date format
set_A["Date"] = pd.to_datetime(set_A["Date"])
set_B["Date"] = pd.to_datetime(set_B["Date"])


# Display the updated sets with the 'Date' column in datetime format
print(set_A.head())
print(set_B.head())

# %%
# Build a concatenated dataset out of both sets
# Subset the data to include only relevant columns
subset_A = set_A[["Date", "Close", "Ticker"]]
subset_B = set_B[["Date", "Close", "Ticker"]]

# %%
# Assuming subset_A and subset_B are DataFrames
combined_data = pd.concat([subset_A, subset_B], ignore_index=True)

# %%
# Check for duplicates in combined_data
duplicates = combined_data[combined_data.duplicated(subset=["Ticker", "Date"])]

# Display the duplicate rows
print(duplicates)

# Count the number of duplicate rows
num_duplicates = len(duplicates)
print("Number of duplicate rows:", num_duplicates)

# %%
# Deduplicate based on 'Date' within each group of 'Ticker'
deduplicated_data = (
    combined_data.groupby("Ticker")
    .apply(lambda group: group.drop_duplicates("Date", keep="first"))
    .reset_index(drop=True)
)

# Display the first few rows of the deduplicated dataset
print(deduplicated_data.head())

# %%
# Create a subset of deduplicated_data with the 500 top tickers by # of observations
ticker_counts = deduplicated_data["Ticker"].value_counts()
top_500_tickers = ticker_counts.head(500).index

combined_data_500 = deduplicated_data[
    deduplicated_data["Ticker"].isin(top_500_tickers)
].copy()

# %%
# Visualize the data in a scatter plot to see the time span for most of the observations in the data
# Create a separate dataframe plot_df_500 based on combined_data_500
plot_df_500 = combined_data_500.copy()

# Calculate the number of observations for each ticker and assign it to a new column 'n'
plot_df_500["n"] = plot_df_500.groupby("Ticker")["Ticker"].transform("count")

# Create the scatter plot with x-axis as Date and y-axis as Ticker
plt.figure(figsize=(16, 16))
plt.scatter(plot_df_500["Date"], plot_df_500["Ticker"], s=1, color="black", alpha=0.5)

# Customize the plot
plt.title("Scatter Plot of Tickers Over Time")
plt.xlabel("Date")
plt.ylabel("Ticker")
plt.xticks(rotation=90)
plt.yticks(fontsize=3)

# Set daily x-axis ticks
plt.gca().xaxis.set_major_locator(DayLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

# Customize the grid to align with x-axis ticks
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Remove the legend
plt.legend().set_visible(False)

# Show the plot
plt.show()

# %%
# A conservative visual assesment would be that the majority of the information is located between Monday 10-10-2022 and Friday 19-11-2022 (only weekdays where trade normally takes place were selected, but the data appears to include after-hours trading until 2am the following day for each trading day).

# %%
# Set the date window
start_date = "2022-10-10"
end_date = "2022-11-19"

# Filter data within the specified date range
data_500_window = combined_data_500[
    (combined_data_500["Date"] >= start_date) & (combined_data_500["Date"] <= end_date)
]

# Organise by Date
data_500_window = data_500_window.sort_values(by=["Ticker", "Date"])

# Display the first few rows of the filtered dataset
print(data_500_window.head())

# %%
# Pivot the data setting missing observations to NA for each Ticker
pivoted_df1 = data_500_window.pivot_table(
    index="Date", columns="Ticker", values="Close", aggfunc="first"
)

print(pivoted_df1.head())

# %%
# Subset the pivoted data resampling it to 3 intervals - minute, half hour and hour
# df_minute = pivoted_df1.resample('1T').last()
# df_Hhour = pivoted_df1.resample('30T').last()
df_hourly = pivoted_df1.resample("H").last()

# %%


# %%
# Subset the dataframe to the best tickers by # of observations
column_counts = df_hourly.count()
sorted_columns = column_counts.sort_values(ascending=False)
top_100_columns = sorted_columns.head(100)
# df_minute_top_100 = df_minute.loc[:, top_100_columns.index]
# df_Hhour_top_100 = df_Hhour.loc[:, top_100_columns.index]
df_hour_top_100 = df_hourly.loc[:, top_100_columns.index]

# %%

df_hour_top_100.index = df_hour_top_100.index.tz_localize("Etc/GMT+1")
df_hour_top_100.index = df_hour_top_100.index.tz_convert("America/New_York")

# Format timestamp without offset
df_hour_top_100.index = df_hour_top_100.index.strftime("%Y-%m-%d %H:%M:%S")
df_hour_top_100.index = pd.to_datetime(df_hour_top_100.index)

# %%
# Define trading hours
start_day_hour = "07:00:00"
end_day_hour = "21:00:00"

# Filter DataFrame to keep observations within trading hours
# df_trading_hours_minute = df_minute_top_100.between_time(start_day_hour, end_day_hour)
# df_trading_hours_Hhour = df_Hhour_top_100.between_time(start_day_hour, end_day_hour)
df_trading_hours_hour = df_hour_top_100.between_time(start_day_hour, end_day_hour)

# %%
# Clean entire rows with NA
# df_trading_hours_minute_cleaned = df_trading_hours_minute.dropna(how='all')
# df_trading_hours_Hhour_cleaned = df_trading_hours_Hhour.dropna(how='all')
df_trading_hours_hour_cleaned = df_trading_hours_hour.dropna(how="all")

# Check for NA values left
# print("Any NA values in df_trading_hours_minute_cleaned:", df_trading_hours_minute_cleaned.isna().any().any())
# print("Any NA values in df_trading_hours_Hhour_cleaned:", df_trading_hours_Hhour_cleaned.isna().any().any())
print(
    "Any NA values in df_trading_hours_hour_cleaned:",
    df_trading_hours_hour_cleaned.isna().any().any(),
)


# %%
# Define the linear interpolation function
def linear_interpolation(series):
    na_index = np.where(series.isna())[0]

    for i in na_index:
        before_candidates = np.where(~series.isna())[0]
        before_candidates = before_candidates[before_candidates < i]

        after_candidates = np.where(~series.isna())[0]
        after_candidates = after_candidates[after_candidates > i]

        if len(before_candidates) > 0 and len(after_candidates) > 0:
            before = max(before_candidates)
            after = min(after_candidates)

            series.iat[i] = series.iat[before] + (
                (series.iat[after] - series.iat[before]) / (after - before)
            ) * (i - before)

    return series


# Apply linear interpolation to df_trading_hours_minute_cleaned
# final_minute = df_trading_hours_minute_cleaned.apply(linear_interpolation, axis=0)

# Apply linear interpolation to df_trading_hours_Hhour_cleaned
# final_Hhour = df_trading_hours_Hhour_cleaned.apply(linear_interpolation, axis=0)

# Apply linear interpolation to df_trading_hours_hour_cleaned
final_hour = df_trading_hours_hour_cleaned.apply(linear_interpolation, axis=0)


# %%
# Drop columns with a large number of preceeding or trailing NAs
# Define a function to drop columns with more than a quarter of NaN values
def drop_columns_with_nan(df, threshold=0.05):
    threshold_count = int(len(df) * threshold)
    nan_counts = df.isnull().sum()
    dropped_columns = df.columns[nan_counts > threshold_count]

    if len(dropped_columns) > 0:
        print(f"Dropping columns: {dropped_columns}")

    cleaned_df = df.loc[:, ~df.columns.isin(dropped_columns)]

    if len(dropped_columns) > 0:
        print(f"Columns after dropping: {cleaned_df.columns}")

    return cleaned_df


# Apply the function to your DataFrames
# final_minute_cleaned = drop_columns_with_nan(final_minute)
# final_Hhour_cleaned = drop_columns_with_nan(final_Hhour)
final_hour_cleaned = drop_columns_with_nan(final_hour)

# %%
# Drop all rows with NaN values
# minute = final_minute_cleaned.dropna()
# half_hour = final_Hhour_cleaned.dropna()
hour = final_hour_cleaned.dropna()

# # Free up some memory
# del (
#     set_A,
#     set_B,
#     pivoted_df1,
#     data_500_window,
#     combined_data_500,
#     subset_A,
#     subset_B,
# )
# del final_hour, final_hour_cleaned
# del df_trading_hours_hour_cleaned

# %%
# Calculate returns for each dataframe
# returns_minute = minute.pct_change().dropna()
# returns_half_hour = half_hour.pct_change().dropna()
returns_hour = hour.pct_change().dropna()

print(returns_hour.head())

# %%
# Create portfolios of equal weights for each dataframe
# ew_port_minute = returns_minute.mean(axis=1)
# ew_port_half_hour = returns_half_hour.mean(axis=1)
ew_port_hour = returns_hour.mean(axis=1)

print(ew_port_hour.head())

# %%
# ### Analysing final_hour
# Calculate the cumulative return for ew_port_hour
cumulative_ew_port_hour = (1 + ew_port_hour).cumprod() - 1

# %%
# Plot the time series of cumulative returns for the equally weighted portfolio of hourly transactions
# Set the style for the plot
sns.set(style="whitegrid")


# Function to format y-axis ticks as percentages
def percentage_formatter(x, pos):
    return f"{100 * x:.2f}%"


# Create a line plot for the cumulative return
plt.figure(figsize=(16, 10))
sns.lineplot(data=cumulative_ew_port_hour, linewidth=2)
plt.title("Cumulative Return of Equal-Weighted Portfolio (Hourly)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")

# Rotate xticks for better visibility
plt.xticks(rotation=45)

# Identify the index of the global minimum and maximum
min_index = cumulative_ew_port_hour.idxmin()
max_index = cumulative_ew_port_hour.idxmax()

# Add another line at the global maximum
plt.axhline(
    y=cumulative_ew_port_hour.loc[max_index], color="green", linestyle="--", label="Max"
)

# Add a red horizontal line at the global minimum
plt.axhline(
    y=cumulative_ew_port_hour.loc[min_index], color="red", linestyle="--", label="Min"
)

# Add text annotations for the maximum and minimum points
plt.text(
    min_index,
    cumulative_ew_port_hour.loc[min_index],
    f"Min: {cumulative_ew_port_hour.loc[min_index]:.4f}",
    color="red",
    ha="right",
    va="top",
)
plt.text(
    max_index,
    cumulative_ew_port_hour.loc[max_index],
    f"Max: {cumulative_ew_port_hour.loc[max_index]:.4f}",
    color="green",
    ha="right",
    va="bottom",
)

# Format y-axis ticks as percentages
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

# Set the number of y-axis ticks
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=60))

# Adjust y-axis tick parameters
plt.yticks(fontsize=6, rotation=45)

# Shade weekends
dates = cumulative_ew_port_hour.index
weekend_mask = dates.weekday >= 5  # Assuming Saturday and Sunday are weekends

for i in range(len(dates) - 1):
    if weekend_mask[i] and not weekend_mask[i + 1]:
        plt.axvspan(
            dates[i], dates[i + 1], facecolor="lightgray", edgecolor="none", alpha=0.5
        )

plt.legend()
plt.show()


# %%
# Function to calculate portfolio return
def portfolio_return(weights, returns):
    return np.dot(weights, np.mean(returns, axis=0))


# Function to calculate portfolio volatility
def portfolio_volatility(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(np.cov(returns.T), weights)))


# Function for the optimization (minimize volatility)
def minimize_volatility(weights, returns):
    return portfolio_volatility(weights, returns)


returns = returns_hour
tickers = returns_hour.columns

# Number of assets
num_assets = len(
    returns_hour.columns
)  # Use length of columns in returns_hour dataframe

# Constraints and bounds
constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for asset in range(num_assets))

# Portfolio optimization for efficient frontier
portfolio_returns = []
portfolio_volatilities = []
portfolio_weights = []

for return_target in np.linspace(returns.mean().min(), returns.mean().max(), 100):
    constraints = (
        {"type": "eq", "fun": lambda x: portfolio_return(x, returns) - return_target},
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
    )
    result = minimize(
        minimize_volatility,
        num_assets
        * [
            1.0 / num_assets,
        ],
        args=(returns,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    portfolio_returns.append(return_target)
    portfolio_volatilities.append(portfolio_volatility(result.x, returns))
    portfolio_weights.append(result.x)

# Find the Minimum Volatility Portfolio (MVP)
mvp_index = np.argmin(portfolio_volatilities)
mvp_return = portfolio_returns[mvp_index]
mvp_volatility = portfolio_volatilities[mvp_index]
mvp_weights = portfolio_weights[mvp_index]

# Simulate 100,000 random portfolios with a normal distribution
num_portfolios = 10000
random_portfolios = []

for _ in range(num_portfolios):
    weights = np.random.normal(
        0, 1, size=num_assets
    )  # Use normal distribution with larger std deviation
    weights /= np.sum(np.abs(weights))  # Ensure weights sum to 1
    return_portfolio = portfolio_return(weights, returns)
    volatility_portfolio = portfolio_volatility(weights, returns)
    random_portfolios.append((return_portfolio, volatility_portfolio, weights))

# Extract returns and volatilities for plotting
random_portfolio_returns, random_portfolio_volatilities, random_portfolio_weights = zip(
    *random_portfolios
)

# %%
returns = returns_hour


# Function to find portfolio weights for a given volatility level
def find_portfolio_weights(volatility_level):
    closest_volatility_idx = np.argmin(
        np.abs(np.array(portfolio_volatilities) - volatility_level)
    )

    return portfolio_weights[closest_volatility_idx - 1]


# User input for a standard deviation level
desired_volatility = round(ew_port_hour.std(), 3)
weights = find_portfolio_weights(desired_volatility)
return_per_risk = portfolio_return(find_portfolio_weights(desired_volatility), returns)
print(f"Return for risk: {return_per_risk:.4f}")
print("Portfolio weights for the desired volatility level:")
for ticker, weight in zip(tickers, weights):
    print(f"{ticker}: {weight:.4f}")

# %%
# Plotting the efficient frontier and random portfolios
plt.figure(figsize=(10, 6))
plt.scatter(
    random_portfolio_volatilities,
    random_portfolio_returns,
    c=random_portfolio_returns,
    cmap="viridis",
    marker="o",
    alpha=0.7,
    label="Random Portfolios",
    edgecolors="k",
    s=10,
)
plt.plot(
    portfolio_volatilities,
    portfolio_returns,
    "b-",
    linewidth=2,
    label="Efficient Frontier",
)
plt.scatter(
    mvp_volatility,
    mvp_return,
    marker="*",
    color="red",
    label="Minimum Volatility Portfolio (MVP)",
)
plt.scatter(
    desired_volatility,
    ew_port_hour.mean(),
    marker="*",
    color="silver",
    label="Equally-Weighted Portfolio",
)
plt.scatter(
    desired_volatility,
    return_per_risk,
    marker="*",
    color="purple",
    label="Optimized Portfolio Equivalent to EW Portfolio",
)

# Display the tuple of (volatility, return) for the MVP
plt.annotate(
    f"MVP: ({mvp_return:.4f}, {mvp_volatility:.4f})",
    (mvp_volatility, mvp_return),
    textcoords="offset points",
    xytext=(-10, -15),
    ha="left",
    fontsize=8,
    color="red",
    backgroundcolor="0.75",
)

# Display the tuple of (volatility, return) for the equally-weighted portfolio
plt.annotate(
    f"EW Portfolio: ({ew_port_hour.mean():.4f}, {desired_volatility:.4f})",
    (desired_volatility, ew_port_hour.mean()),
    textcoords="offset points",
    xytext=(10, 10),
    ha="left",
    fontsize=8,
    color="black",
    backgroundcolor="0.75",
)

# Display the tuple of (volatility, return) for the optimized portfolio with risk equal to ew portfolio
plt.annotate(
    f"Optimized Equivalent: ({return_per_risk:.4f}, {desired_volatility:.4f})",
    (desired_volatility, return_per_risk),
    textcoords="offset points",
    xytext=(10, 10),
    ha="left",
    fontsize=8,
    color="purple",
    backgroundcolor="0.75",
)

plt.title("Efficient Frontier with Random Portfolios and MVP (Normal Distribution)")
plt.xlabel("Portfolio Volatility (Standard Deviation)")
plt.ylabel("Portfolio Return")
plt.legend()

# Set axis limits
plt.ylim(-0.002, 0.004)
plt.xlim(0.00, 0.02)

plt.show()

# %%
# # Clean memory
# del (
#     plot_df_500,
#     random_portfolio_returns,
#     random_portfolio_volatilities,
#     random_portfolio_weights,
# )
# gc.collect()
# %%
####################### Rolling Window #######################


# Function to calculate portfolio return
def portfolio_return(weights, returns):
    return np.dot(weights, np.mean(returns, axis=0))


# Function to calculate portfolio volatility
def portfolio_volatility(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(np.cov(returns.T), weights)))


# Function for the optimization (minimize volatility)
def minimize_volatility(weights, returns):
    return portfolio_volatility(weights, returns)


# Number of assets
num_assets = len(returns.columns)

# Rolling window size
rolling_window_size = 24  # Adjust as needed

# Initialize an empty DataFrame to store the results
optimized_weights_df_24 = pd.DataFrame(index=returns.index, columns=returns.columns)

# Loop through the rolling window
for i in range(len(returns) - rolling_window_size + 1):
    window_returns = returns.iloc[i : i + rolling_window_size, :]

    # Constraints and bounds
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Initial weights for the optimization
    initial_weights = np.ones(num_assets) / num_assets

    # Perform optimization
    result = minimize(
        minimize_volatility,
        initial_weights,
        args=(window_returns,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    # Store optimized weights in the DataFrame
    optimized_weights_df_24.iloc[i + rolling_window_size - 1, :] = result.x

optimized_weights_df_24 = optimized_weights_df_24.dropna()

# Print the DataFrame with optimized weights
print(optimized_weights_df_24)

# %%
optimized_returns_24_window = pd.DataFrame(
    index=optimized_weights_df_24.index, columns=["Return"]
)

returns_24_window = returns.loc[optimized_weights_df_24.index]
print(returns_24_window)

weights = optimized_weights_df_24.iloc[i, :].values

optimized_returns_24_window = optimized_weights_df_24.apply(
    lambda weights: portfolio_return(weights, returns_24_window), axis=1
)

optimized_returns_24_window

# %%
# Calculate the cumulative return for optimized_returns_24_window
cumulative_optimized_returns_24_window = (1 + optimized_returns_24_window).cumprod() - 1

# %%
# Plot the time series of cumulative returns for the equally weighted portfolio of hourly transactions
# Set the style for the plot
sns.set(style="whitegrid")


# Function to format y-axis ticks as percentages
def percentage_formatter(x, pos):
    return f"{100 * x:.2f}%"


# Create a line plot for the cumulative return
plt.figure(figsize=(16, 10))
sns.lineplot(
    data=cumulative_ew_port_hour,
    linewidth=2,
    color="green",
    label="Equal-Weighted Portfolio (Hourly)",
)
sns.lineplot(
    data=cumulative_optimized_returns_24_window,
    linewidth=2,
    color="purple",
    label="Optimized Portfolio of Equivalent Risk (24-Periods)",
)
plt.title(
    "Cumulative Return of Equal-Weighted Portfolio (Hourly) vs. Optimized Portfolio of Equivalent Risk"
)
plt.xlabel("Date")
plt.ylabel("Cumulative Return")

# Rotate xticks for better visibility
plt.xticks(rotation=45)

# Identify the index of the global minimum and maximum
min_index = cumulative_ew_port_hour.idxmin()
max_index = cumulative_ew_port_hour.idxmax()

# Add another line at the global maximum
plt.axhline(
    y=cumulative_ew_port_hour.loc[max_index], color="black", linestyle="--", label="Max"
)

# Add a red horizontal line at the global minimum
plt.axhline(
    y=cumulative_ew_port_hour.loc[min_index], color="red", linestyle="--", label="Min"
)

# Add text annotations
plt.text(
    min_index,
    cumulative_ew_port_hour.loc[min_index],
    f"Min: {cumulative_ew_port_hour.loc[min_index]*100:.2f}%",
    color="red",
    ha="right",
    va="top",
)
plt.text(
    max_index,
    cumulative_ew_port_hour.loc[max_index],
    f"Max: {cumulative_ew_port_hour.loc[max_index]*100:.2f}%",
    color="black",
    ha="right",
    va="bottom",
)

# Annotate points on the line plots
plt.annotate(
    f"Optimized Min: {cumulative_optimized_returns_24_window.min()*100:.2f}%",
    xy=(
        cumulative_optimized_returns_24_window.idxmin(),
        cumulative_optimized_returns_24_window.min(),
    ),
    xytext=(15, 10),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="purple"),
    color="purple",
)

plt.annotate(
    f"Optimized Max: {cumulative_optimized_returns_24_window.max()*100:.2f}%",
    xy=(
        cumulative_optimized_returns_24_window.idxmax(),
        cumulative_optimized_returns_24_window.max(),
    ),
    xytext=(15, -10),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="purple"),
    color="purple",
)

# Format y-axis ticks as percentages
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

# Set the number of y-axis ticks
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=60))

# Adjust y-axis tick parameters
plt.yticks(fontsize=6, rotation=45)

# Shade weekends
dates = cumulative_ew_port_hour.index
weekend_mask = dates.weekday >= 5  # Assuming Saturday and Sunday are weekends

for i in range(len(dates) - 1):
    if weekend_mask[i] and not weekend_mask[i + 1]:
        plt.axvspan(
            dates[i], dates[i + 1], facecolor="lightgray", edgecolor="none", alpha=0.5
        )

plt.legend()
plt.show()

# %%
############################### CAPM (Fama-French) MODEL ###############################
# Extract start date and end date with hour
start_date = optimized_returns_24_window.index[0].date()
end_date = optimized_returns_24_window.index[-1].date() + pd.Timedelta(days=1)

ff = pd.read_csv("data/F-F_Research_Data_5_Factors_2x3_daily.csv")
ff["Date"] = pd.to_datetime(
    ff["Date"], format="%d/%m/%Y"
)  # Convert 'Date' to datetime format without changing the order

# Set 'Date' as the index
ff.set_index("Date", inplace=True)

# Subset the Fama-French data between start_date and end_date
subset_ff = ff.loc[start_date:end_date]
subset_ff = subset_ff.div(100)

# Display the subset
print(subset_ff)

# %%
# Resample 'subset_ff' to hourly frequency with forward-fill
# Linearly interpolate 'subset_ff' to hourly frequency
subset_ff_hourly = subset_ff.resample("H").interpolate(method="linear")

subset_ff_hourly = subset_ff_hourly.loc[optimized_returns_24_window.index]

# Divide by the number of observations per day to distribute the daily values
subset_ff_hourly = subset_ff_hourly / subset_ff_hourly.groupby(
    subset_ff_hourly.index.date
).transform("count")

# Display the resulting dataframe
print(subset_ff_hourly)
# %%
# Run a Fama-French 3-factor model
# Prepare the data
excess_returns = optimized_returns_24_window - subset_ff_hourly["RF"]
X = sm.add_constant(subset_ff_hourly[["Mkt-RF", "SMB", "HML"]])
y = excess_returns

# Fit the regression model
model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

# Display the regression results
print(model.summary())
# %%
