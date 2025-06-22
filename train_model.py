# enhanced_model.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath='synthetic_fraud_data.csv'):
    # Load your dataset
    df = pd.read_csv(filepath)
    
    # Ensure all expected columns are present
    expected_cols = [
        'amount', 'sender_balance', 'receiver_balance', 'transaction_speed',
        'sender_country', 'receiver_country', 'sender_id', 'receiver_id',
        'transaction_frequency', 'is_cross_border', 'high_risk_sender',
        'high_risk_receiver', 'time_since_last_tx', 'is_fraud',
        'amount_to_balance_ratio'
    ]
    
    for col in expected_cols:
        if col not in df.columns:
            if col == 'amount_to_balance_ratio':
                df['amount_to_balance_ratio'] = df['amount'] / df['sender_balance']
            elif col == 'is_cross_border':
                df['is_cross_border'] = (df['sender_country'] != df['receiver_country']).astype(int)
    
    return df

def preprocess_data(df):
    # Drop ID columns
    df = df.drop(['sender_id', 'receiver_id'], axis=1)
    
    # Convert categorical features
    df = pd.get_dummies(df, columns=['sender_country', 'receiver_country'])
    
    # Convert boolean to int
    bool_cols = ['is_cross_border', 'high_risk_sender', 'high_risk_receiver']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    return df


def train_model():
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Separate features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train XGBoost model with optimized parameters
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\nFeature Importances:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance.head(15))
    
    # Save artifacts
    joblib.dump(model, 'models/enhanced_fraud_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    
    return model

if __name__ == "__main__":
    train_model()