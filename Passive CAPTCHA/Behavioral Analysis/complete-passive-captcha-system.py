import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Data Generation Functions
def simulate_human_data(n_samples=1000):
    return pd.DataFrame({
        'keystroke_mean_hold': np.random.normal(0.1, 0.02, n_samples),
        'keystroke_std_hold': np.random.normal(0.05, 0.01, n_samples),
        'keystroke_mean_flight': np.random.normal(0.2, 0.05, n_samples),
        'keystroke_std_flight': np.random.normal(0.1, 0.02, n_samples),
        'typing_speed': np.random.normal(5, 1, n_samples),
        'mouse_mean_speed': np.random.normal(200, 50, n_samples),
        'mouse_std_speed': np.random.normal(50, 10, n_samples),
        'mouse_mean_accel': np.random.normal(1000, 200, n_samples),
        'mouse_std_accel': np.random.normal(500, 100, n_samples),
        'mouse_direction_changes': np.random.poisson(20, n_samples),
        'is_human': 1
    })

def simulate_bot_data(n_samples=1000):
    return pd.DataFrame({
        'keystroke_mean_hold': np.random.normal(0.05, 0.01, n_samples),
        'keystroke_std_hold': np.random.normal(0.01, 0.005, n_samples),
        'keystroke_mean_flight': np.random.normal(0.1, 0.02, n_samples),
        'keystroke_std_flight': np.random.normal(0.02, 0.005, n_samples),
        'typing_speed': np.random.normal(10, 0.5, n_samples),
        'mouse_mean_speed': np.random.normal(500, 20, n_samples),
        'mouse_std_speed': np.random.normal(10, 2, n_samples),
        'mouse_mean_accel': np.random.normal(2000, 100, n_samples),
        'mouse_std_accel': np.random.normal(100, 20, n_samples),
        'mouse_direction_changes': np.random.poisson(5, n_samples),
        'is_human': 0
    })

# Model Training and Evaluation Functions
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    dt_model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    
    print("Best parameters:", grid_search.best_params_)
    
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': best_model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    return best_model, scaler

# Prediction Functions
def predict_interaction(model, scaler, features):
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    return "Human" if prediction[0] == 1 else "Bot", probabilities[0]

def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")

def manual_prediction(model, scaler):
    print("Enter the following features for the interaction:")
    
    features = []
    feature_names = [
        "Keystroke Mean Hold Time (seconds)",
        "Keystroke Std Dev Hold Time",
        "Keystroke Mean Flight Time (seconds)",
        "Keystroke Std Dev Flight Time",
        "Typing Speed (characters per second)",
        "Mouse Mean Speed (pixels per second)",
        "Mouse Std Dev Speed",
        "Mouse Mean Acceleration (pixels per second^2)",
        "Mouse Std Dev Acceleration",
        "Mouse Direction Changes (count)"
    ]

    for name in feature_names:
        value = get_float_input(f"{name}: ")
        features.append(value)

    features = np.array(features)

    prediction, probabilities = predict_interaction(model, scaler, features)

    print(f"\nPrediction: {prediction}")
    print(f"Probability of being a Bot: {probabilities[0]:.4f}")
    print(f"Probability of being a Human: {probabilities[1]:.4f}")

# Main execution
def main():
    # Generate and combine data
    print("Generating data...")
    human_data = simulate_human_data(5000)
    bot_data = simulate_bot_data(5000)
    combined_data = pd.concat([human_data, bot_data], ignore_index=True)
    combined_data = combined_data.sample(frac=1).reset_index(drop=True)

    # Save data to CSV
    combined_data.to_csv('passive_captcha_data.csv', index=False)
    print("Data saved to 'passive_captcha_data.csv'")

    # Prepare features and labels
    X = combined_data.drop('is_human', axis=1)
    y = combined_data['is_human']

    # Train model
    print("\nTraining model...")
    model, scaler = train_model(X, y)

    # Save model and scaler
    joblib.dump(model, 'passive_captcha_model.joblib')
    joblib.dump(scaler, 'passive_captcha_scaler.joblib')
    print("Model and scaler saved.")

    # Prediction loop
    while True:
        print("\nChoose an option:")
        print("1. Make a prediction with manual input")
        print("2. Generate and predict bot data")
        print("3. Exit")
        
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            manual_prediction(model, scaler)
        elif choice == '2':
            bot_data = simulate_bot_data(100)
            bot_data_scaled = scaler.transform(bot_data.drop('is_human', axis=1))
            predictions = model.predict(bot_data_scaled)
            accuracy = (predictions == 0).mean()
            print(f"\nPercentage of samples classified as bots: {accuracy*100:.2f}%")
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

    print("Thank you for using the Passive CAPTCHA system!")

if __name__ == "__main__":
    main()
