import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight, resample
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Input, LayerNormalization, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.stats import ttest_ind, ttest_1samp

# Fetch S&P 500 historical data up to the current date and resample to weekly frequency
# ticker = "GC=F"
# ticker = "BTC-USD"
# ticker = "^GSPC"
# ticker = "AAPL"
ticker = "^VIX"
end_date = pd.to_datetime("2024-10-30")
data = yf.download(ticker, start="1936-01-01", end=end_date.strftime('%Y-%m-%d'))

# Resample the data to weekly frequency (week ending on Friday)
# data = data.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

# Calculate returns and create labels for weekly data
data['Return'] = data['Close'].pct_change()
data['Label'] = (data['Return'] > 0).astype(int)

# Drop NaN values
data = data.dropna()

# Function to create OHLC images from weekly data
def create_ohlc_image(df, window=52):
    images = []
    labels = []
    dates = []
    for i in range(window, len(df)):
        window_data = df.iloc[i-window:i]
        ohlc_image = window_data[['Open', 'High', 'Low', 'Close']].values
        ohlc_image = MinMaxScaler().fit_transform(ohlc_image)  # Normalize data
        images.append(ohlc_image)
        labels.append(df.iloc[i]['Label'])
        dates.append(df.index[i])
    return np.array(images), np.array(labels), np.array(dates)

# Function to split data into training, validation, and testing sets
def split_data(data, train_size=0.6, val_size=0.2):
    train_idx = int(len(data) * train_size)
    val_idx = int(len(data) * (train_size + val_size))

    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]

    return train_data, val_data, test_data

# Split the data
train_data, val_data, test_data = split_data(data)

# Create OHLC images based on weekly data
X_train, y_train, dates_train = create_ohlc_image(train_data)
X_val, y_val, dates_val = create_ohlc_image(val_data)
X_test, y_test, dates_test = create_ohlc_image(test_data)

# Oversample the minority class in the training set
def oversample_minority_class(X, y):
    X_majority = X[y == 1]
    X_minority = X[y == 0]
    y_majority = y[y == 1]
    y_minority = y[y == 0]

    # Oversample minority class
    X_minority_oversampled, y_minority_oversampled = resample(
        X_minority, y_minority,
        replace=True,    # sample with replacement
        n_samples=len(y_majority),  # match number in majority class
        random_state=42  # reproducible results
    )

    # Combine majority and oversampled minority class
    X_oversampled = np.vstack((X_majority, X_minority_oversampled))
    y_oversampled = np.hstack((y_majority, y_minority_oversampled))

    return X_oversampled, y_oversampled

# Apply oversampling before reshaping for CNN, LSTM, ViT, and CNN-LSTM
X_train, y_train = oversample_minority_class(X_train, y_train)

# Reshape data to fit the CNN input (4D: samples, timesteps, features, 1)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 4, 1)
X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 4, 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 4, 1)

# Reshape data to fit the LSTM input (3D: samples, timesteps, features)
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)
X_val_lstm = X_val.reshape(X_val.shape[0], X_val.shape[1], 4)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 4)

# Reshape data to fit the ViT input (3D: samples, timesteps, features)
X_train_vit = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)
X_val_vit = X_val.reshape(X_val.shape[0], X_val.shape[1], 4)
X_test_vit = X_test.reshape(X_test.shape[0], X_test.shape[1], 4)

# Reshape data for CNN-LSTM (5D for CNN-LSTM: samples, timesteps, height, width, channels)
X_train_cnn_lstm = X_train_cnn.reshape(X_train_cnn.shape[0], X_train_cnn.shape[1], 1, 4, 1)
X_val_cnn_lstm = X_val_cnn.reshape(X_val_cnn.shape[0], X_val_cnn.shape[1], 1, 4, 1)
X_test_cnn_lstm = X_test_cnn.reshape(X_test_cnn.shape[0], X_test_cnn.shape[1], 1, 4, 1)

# Define CNN model
def build_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 1), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 1)),
        Conv2D(64, (3, 1), activation='relu', padding='same'),
        MaxPooling2D((2, 1)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define CNN-LSTM model
def build_cnn_lstm_model(input_shape):
    model = Sequential()

    # CNN layers to extract features
    model.add(TimeDistributed(Conv2D(32, (1, 1), activation='relu', padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((1, 1))))
    model.add(TimeDistributed(Conv2D(64, (1, 1), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((1, 1))))
    model.add(TimeDistributed(Flatten()))

    # LSTM layer to handle temporal dependencies
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define ViT model
def build_vit_model(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)
    x = LayerNormalization()(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build CNN, LSTM, ViT, and CNN-LSTM models
cnn_model = build_cnn_model(input_shape=(X_train_cnn.shape[1], 4, 1))
lstm_model = build_lstm_model(input_shape=(X_train_lstm.shape[1], 4))
cnn_lstm_model = build_cnn_lstm_model(input_shape=(X_train_cnn_lstm.shape[1], X_train_cnn_lstm.shape[2], X_train_cnn_lstm.shape[3], X_train_cnn_lstm.shape[4]))
vit_model = build_vit_model(input_shape=(X_train_vit.shape[1], 4))

# Compute class weights to handle class imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Train the CNN, LSTM, ViT, and CNN-LSTM models
early_stopping_cnn = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_cnn = cnn_model.fit(X_train_cnn, y_train, epochs=30, batch_size=32, validation_data=(X_val_cnn, y_val),
                            callbacks=[early_stopping_cnn], class_weight=class_weights_dict)

early_stopping_lstm = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_lstm = lstm_model.fit(X_train_lstm, y_train, epochs=30, batch_size=32, validation_data=(X_val_lstm, y_val),
                              callbacks=[early_stopping_lstm], class_weight=class_weights_dict)

early_stopping_cnn_lstm = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_cnn_lstm = cnn_lstm_model.fit(X_train_cnn_lstm, y_train, epochs=30, batch_size=32, validation_data=(X_val_cnn_lstm, y_val),
                                      callbacks=[early_stopping_cnn_lstm], class_weight=class_weights_dict)

early_stopping_vit = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_vit = vit_model.fit(X_train_vit, y_train, epochs=30, batch_size=32, validation_data=(X_val_vit, y_val),
                            callbacks=[early_stopping_vit], class_weight=class_weights_dict)

# Evaluate the models
loss_cnn, accuracy_cnn = cnn_model.evaluate(X_test_cnn, y_test)
loss_lstm, accuracy_lstm = lstm_model.evaluate(X_test_lstm, y_test)
loss_cnn_lstm, accuracy_cnn_lstm = cnn_lstm_model.evaluate(X_test_cnn_lstm, y_test)
loss_vit, accuracy_vit = vit_model.evaluate(X_test_vit, y_test)

print(f'CNN Test Accuracy: {accuracy_cnn:.2f}')
print(f'LSTM Test Accuracy: {accuracy_lstm:.2f}')
print(f'CNN-LSTM Test Accuracy: {accuracy_cnn_lstm:.2f}')
print(f'ViT Test Accuracy: {accuracy_vit:.2f}')

# Implement the investment strategy on a weekly basis
def investment_strategy(model, X_test, dates_test, model_name):
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate cumulative returns
    actual_cum_returns = [1]
    strategy_cum_returns = [1]

    strategy_position = 0  # Start with no position
    for i in range(len(y_pred)):
        date = dates_test[i]
        if date in data.index:
            actual_return = data.loc[date, 'Return']
        else:
            nearest_idx = data.index.get_indexer([date], method='nearest')[0]
            actual_return = data.iloc[nearest_idx]['Return']

        if y_pred[i] == 1:  # Buy 1 share when predicted return is positive
            strategy_position = 1.5
        else:
            strategy_position = 0.5 # Do nothing

        strategy_return = strategy_position * actual_return

        actual_cum_returns.append(actual_cum_returns[-1] * (1 + actual_return))
        strategy_cum_returns.append(strategy_cum_returns[-1] * (1 + strategy_return))

    # Remove the initial value (1)
    actual_cum_returns = actual_cum_returns[1:]
    strategy_cum_returns = strategy_cum_returns[1:]

    # Convert to percentage returns for easier comparison
    actual_cum_returns = np.array(actual_cum_returns) - 1
    strategy_cum_returns = np.array(strategy_cum_returns) - 1

    return actual_cum_returns, strategy_cum_returns, dates_test[:len(actual_cum_returns)]

# Investment strategies for CNN, LSTM, ViT, and CNN-LSTM models
actual_cum_returns_cnn, strategy_cum_returns_cnn, dates_cnn = investment_strategy(cnn_model, X_test_cnn, dates_test, 'CNN')
actual_cum_returns_lstm, strategy_cum_returns_lstm, dates_lstm = investment_strategy(lstm_model, X_test_lstm, dates_test, 'LSTM')
actual_cum_returns_vit, strategy_cum_returns_vit, dates_vit = investment_strategy(vit_model, X_test_vit, dates_test, 'ViT')
actual_cum_returns_cnn_lstm, strategy_cum_returns_cnn_lstm, dates_cnn_lstm = investment_strategy(cnn_lstm_model, X_test_cnn_lstm, dates_test, 'CNN-LSTM')

# Function to calculate excess returns
def calculate_excess_returns(cumulative_returns_df):
    strategies = ['CNN_Cum_Return', 'LSTM_Cum_Return', 'ViT_Cum_Return', 'CNN-LSTM_Cum_Return']
    for strategy in strategies:
        cumulative_returns_df[f'{strategy}_Excess'] = (
            cumulative_returns_df[strategy] - cumulative_returns_df['Actual_Cum_Return']
        )
    return cumulative_returns_df

# Function to calculate annualized return and volatility
def calculate_annualized_metrics(cumulative_returns_df, weekly_returns):
    metrics = pd.DataFrame(index=['Annualized Return', 'Annualized Volatility'], columns=['CNN', 'LSTM', 'ViT', 'CNN-LSTM', 'Actual'])
    n_years = len(weekly_returns) / 52  # Number of years in the data

    strategies = {
        'CNN': 'CNN_Cum_Return',
        'LSTM': 'LSTM_Cum_Return',
        'ViT': 'ViT_Cum_Return',
        'CNN-LSTM': 'CNN-LSTM_Cum_Return',
        'Actual': 'Actual_Cum_Return'
    }

    for strategy, column in strategies.items():
        # Annualized Return
        metrics.loc['Annualized Return', strategy] = (1 + cumulative_returns_df[column].iloc[-1]) ** (1 / n_years) - 1

        # Annualized Volatility
        weekly_strategy_returns = weekly_returns[column]
        metrics.loc['Annualized Volatility', strategy] = np.std(weekly_strategy_returns) * np.sqrt(52)

    return metrics

# Function to perform one-sample t-tests for excess returns
def one_sample_t_tests_excess(cumulative_returns_df):
    strategies = ['CNN_Cum_Return_Excess', 'LSTM_Cum_Return_Excess', 'ViT_Cum_Return_Excess', 'CNN-LSTM_Cum_Return_Excess']
    t_test_results = pd.DataFrame(index=strategies, columns=['Mean', 'Std', 't-Stat', 'p-Value'])

    # Perform one-sample t-tests for excess returns against 0
    for strategy in strategies:
        excess_returns = cumulative_returns_df[strategy].dropna()
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()
        n = len(excess_returns)
        t_stat = mean_excess / (std_excess / np.sqrt(n))  # Manual t-stat calculation
        _, p_val = ttest_1samp(excess_returns, popmean=0)

        t_test_results.loc[strategy, 'Mean'] = mean_excess
        t_test_results.loc[strategy, 'Std'] = std_excess
        t_test_results.loc[strategy, 't-Stat'] = t_stat
        t_test_results.loc[strategy, 'p-Value'] = p_val

    return t_test_results

# Calculate cumulative returns and excess returns for each strategy
cumulative_returns_df = pd.DataFrame({
    'Date': dates_test[:len(actual_cum_returns_cnn)],
    'Actual_Cum_Return': actual_cum_returns_cnn,
    'CNN_Cum_Return': strategy_cum_returns_cnn,
    'LSTM_Cum_Return': strategy_cum_returns_lstm,
    'ViT_Cum_Return': strategy_cum_returns_vit,
    'CNN-LSTM_Cum_Return': strategy_cum_returns_cnn_lstm
})

# Calculate excess returns for each strategy
cumulative_returns_df = calculate_excess_returns(cumulative_returns_df)

# Example of weekly returns
weekly_returns = pd.DataFrame({
    'Actual_Cum_Return': cumulative_returns_df['Actual_Cum_Return'].diff(),
    'CNN_Cum_Return': cumulative_returns_df['CNN_Cum_Return'].diff(),
    'LSTM_Cum_Return': cumulative_returns_df['LSTM_Cum_Return'].diff(),
    'ViT_Cum_Return': cumulative_returns_df['ViT_Cum_Return'].diff(),
    'CNN-LSTM_Cum_Return': cumulative_returns_df['CNN-LSTM_Cum_Return'].diff()
}).dropna()

# Calculate annualized return and volatility
annualized_metrics = calculate_annualized_metrics(cumulative_returns_df, weekly_returns)
print("Annualized Metrics (Return and Volatility):")
print(annualized_metrics)

# Perform one-sample t-tests on excess returns
t_test_results_excess = one_sample_t_tests_excess(cumulative_returns_df)
print("\nT-Test Results for Excess Returns (vs 0):")
print(t_test_results_excess)

# Save the results to CSV files
cumulative_returns_df.to_csv('cumulative_excess_returns.csv', index=False)
annualized_metrics.to_csv('annualized_metrics.csv')
t_test_results_excess.to_csv('t_test_results_excess.csv', index=True)

# Plot the cumulative returns for CNN, LSTM, ViT, CNN-LSTM, and Actual returns
plt.figure(figsize=(12, 6))
plt.plot(dates_cnn, actual_cum_returns_cnn, label='Actual Cumulative Return', alpha=0.7)
plt.plot(dates_cnn, strategy_cum_returns_cnn, label='CNN Strategy Cumulative Return', alpha=0.7)
plt.plot(dates_lstm, strategy_cum_returns_lstm, label='LSTM Strategy Cumulative Return', alpha=0.7)
plt.plot(dates_vit, strategy_cum_returns_vit, label='ViT Strategy Cumulative Return', alpha=0.7)
plt.plot(dates_cnn_lstm, strategy_cum_returns_cnn_lstm, label='CNN-LSTM Strategy Cumulative Return', alpha=0.7)
plt.title('Cumulative Returns (Weekly): CNN vs LSTM vs ViT vs CNN-LSTM vs Actual')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()
