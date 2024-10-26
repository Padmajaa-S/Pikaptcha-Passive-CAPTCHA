

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore
from scipy.stats import entropy
import webbrowser

def generate_keystroke_data(num_samples, is_human=True):
    data = []
    for _ in range(num_samples):
        sequence_length = np.random.randint(20, 100)  # Longer sequences for more features

        # Basic timing features
        keystroke_times = []
        pause_times = []
        key_hold_times = []
        key_distances = []
        errors_made = 0
        corrections_made = 0
        copy_paste_events = 0

        for i in range(sequence_length):
            if is_human:
                keystroke_time = np.random.normal(0.2, 0.05)
                pause_time = np.random.normal(0.5, 0.2)
                key_hold_time = np.random.normal(0.1, 0.03)
                key_distance = np.random.normal(2, 1)
            else:
                keystroke_time = np.random.normal(0.05, 0.01)
                pause_time = np.random.normal(0.1, 0.05)
                key_hold_time = np.random.normal(0.05, 0.01)
                key_distance = np.random.normal(1.5, 0.5)

            keystroke_times.append(keystroke_time)
            pause_times.append(pause_time)
            key_hold_times.append(key_hold_time)
            key_distances.append(key_distance)

            if is_human and np.random.random() < 0.05:
                errors_made += 1
                if np.random.random() < 0.8:
                    corrections_made += 1

            if (not is_human and np.random.random() < 0.1) or (is_human and np.random.random() < 0.02):
                copy_paste_events += 1

        avg_keystroke_time = np.mean(keystroke_times)
        std_keystroke_time = np.std(keystroke_times)
        avg_pause_time = np.mean(pause_times)
        std_pause_time = np.std(pause_times)
        avg_key_hold_time = np.mean(key_hold_times)
        std_key_hold_time = np.std(key_hold_times)
        typing_speed = len(keystroke_times) / sum(pause_times)
        rhythm_consistency = entropy(keystroke_times)
        avg_key_distance = np.mean(key_distances)
        std_key_distance = np.std(key_distances)
        error_rate = errors_made / sequence_length
        correction_rate = corrections_made / max(errors_made, 1)
        copy_paste_frequency = copy_paste_events / sequence_length

        mouse_speed = np.random.normal(500, 100) if is_human else np.random.normal(800, 50)
        mouse_acceleration = np.random.normal(200, 50) if is_human else np.random.normal(100, 20)
        mouse_jerk = np.random.normal(100, 30) if is_human else np.random.normal(50, 10)

        features = [
            avg_keystroke_time, std_keystroke_time, avg_pause_time, std_pause_time,
            avg_key_hold_time, std_key_hold_time, typing_speed, rhythm_consistency,
            avg_key_distance, std_key_distance, error_rate, correction_rate,
            copy_paste_frequency, mouse_speed, mouse_acceleration, mouse_jerk
        ]

        data.append(features)

    return data

# Generate data
num_human_samples = 15000
num_bot_samples = 15000

human_data = generate_keystroke_data(num_human_samples, is_human=True)
bot_data = generate_keystroke_data(num_bot_samples, is_human=False)

columns = [
    'avg_keystroke_time', 'std_keystroke_time', 'avg_pause_time', 'std_pause_time',
    'avg_key_hold_time', 'std_key_hold_time', 'typing_speed', 'rhythm_consistency',
    'avg_key_distance', 'std_key_distance', 'error_rate', 'correction_rate',
    'copy_paste_frequency', 'mouse_speed', 'mouse_acceleration', 'mouse_jerk'
]

df_human = pd.DataFrame(human_data, columns=columns)
df_bot = pd.DataFrame(bot_data, columns=columns)

df_human['target'] = 'human'
df_bot['target'] = 'bot'

df = pd.concat([df_human, df_bot], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data

df['target'] = (df['target'] == 'bot').astype(int)

# Clean data
def clean_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    problematic_columns = df.columns[df.isin([np.inf, -np.inf, np.nan]).any()].tolist()

    for col in problematic_columns:
        median_value = df[col].median()
        df[col] = df[col].replace([np.inf, -np.inf, np.nan], median_value)

    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            lower_bound = df[column].quantile(0.001)
            upper_bound = df[column].quantile(0.999)
            df[column] = df[column].clip(lower_bound, upper_bound)

    return df

df = clean_data(df)

# Prepare features and target
X = df.drop('target', axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM input
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build LSTM model
model = Sequential([
    LSTM(128, input_shape=(1, X_train_reshaped.shape[2]), return_sequences=True, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Predict if the input is from a bot or a human
def predict_bot(new_data):
    if new_data.ndim == 1:
        new_data = new_data.reshape(1, -1)

    new_data_scaled = scaler.transform(new_data)
    new_data_reshaped = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))

    prediction = model.predict(new_data_reshaped)
    is_bot = prediction > 0.5
    confidence = prediction if is_bot else 1 - prediction

    return is_bot[0][0], confidence[0][0]

# Example prediction
real_input = np.array([0.2, 0.05, 0.5, 0.1, 0.1, 0.02, 5.0, 0.8, 2.0, 0.5, 0.01, 0.005, 0.001, 300, 100, 50])
is_bot, confidence = predict_bot(real_input)

print(f"Is bot: {is_bot}")

# Open loading.html if it's not a bot
if not is_bot:
    webbrowser.open('loading.html')
else:
    webbrowser.open('Main_gh.html')
