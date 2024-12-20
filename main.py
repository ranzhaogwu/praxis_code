import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Fetch S&P 500 historical data
ticker = "^GSPC"
data = yf.download(ticker, start="1936-01-01", end="2024-06-30")

# Calculate returns and create labels
data['Return'] = data['Close'].pct_change()
data['Label'] = (data['Return'] > 0).astype(int)

# Drop NaN values
data = data.dropna()


def create_ohlc_image(df, window=20):
    images = []
    labels = []
    dates = []
    for i in range(window, len(df)):
        window_data = df.iloc[i - window:i]
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

# Create OHLC images
X_train, y_train, dates_train = create_ohlc_image(train_data)
X_val, y_val, dates_val = create_ohlc_image(val_data)
X_test, y_test, dates_test = create_ohlc_image(test_data)

# Reshape data to fit the CNN input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 4, 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 4, 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 4, 1))

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 1), activation='relu', input_shape=(X_train.shape[1], 4, 1), padding='same'),
    MaxPooling2D((2, 1)),
    Conv2D(64, (3, 1), activation='relu', padding='same'),
    MaxPooling2D((2, 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')


# Plot training & validation accuracy values
def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{title} Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


plot_history(history, 'Weekly')


# Implement the investment strategy
def investment_strategy(model, X_test, y_test, dates_test):
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
            strategy_position = 1
        else:
            strategy_position = 0  # Do nothing

        strategy_return = strategy_position * actual_return

        actual_cum_returns.append(actual_cum_returns[-1] * (1 + actual_return))
        strategy_cum_returns.append(strategy_cum_returns[-1] * (1 + strategy_return))

    # Remove the initial value (1)
    actual_cum_returns = actual_cum_returns[1:]
    strategy_cum_returns = strategy_cum_returns[1:]

    # Convert to percentage returns for easier comparison
    actual_cum_returns = np.array(actual_cum_returns) - 1
    strategy_cum_returns = np.array(strategy_cum_returns) - 1

    # Plot the cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test[:len(actual_cum_returns)], actual_cum_returns, label='Actual Cumulative Return', alpha=0.7)
    plt.plot(dates_test[:len(strategy_cum_returns)], strategy_cum_returns, label='Strategy Cumulative Return',
             alpha=0.7)
    plt.title('Cumulative Returns: Strategy vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.show()


investment_strategy(model, X_test, y_test, dates_test)
