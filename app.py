from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import hashlib
import json
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
import bcrypt


app = Flask(__name__)

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Load models
try:
    model = joblib.load('models/enhanced_fraud_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    # Create dummy models for development
    model = None
    scaler = None
    feature_names = []

 # Database and login setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this!
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), unique=True)
    password = db.Column(db.String(100))
    role = db.Column(db.String(20))  # 'admin', 'analyst', 'auditor'

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(previous_hash="0", data={"system": "Initialized"})

    def create_block(self, previous_hash, data):
        # Recursively convert non-serializable types (e.g., float32) to native Python types
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return obj

        data = convert(data)  # Apply conversion

        block = {
            "index": len(self.chain) + 1,
            "timestamp": str(datetime.now()),
            "data": data,
            "previous_hash": previous_hash,
            "hash": self.hash_block(previous_hash, data)
        }
        self.chain.append(block)
        return block

    def hash_block(self, previous_hash, data):
        block_content = json.dumps(data, sort_keys=True) + previous_hash
        return hashlib.sha256(block_content.encode()).hexdigest()

    def get_last_block(self):
        return self.chain[-1] if self.chain else None


blockchain = Blockchain()

class VTACAnalyzer:
    @staticmethod
    def analyze_transaction(transaction):
        # Phase 1: Detection
        detection_result = VTACAnalyzer.detection_phase(transaction)
        
        # Phase 2: Identification
        identification_result = VTACAnalyzer.identification_phase(transaction)
        
        return {
            **detection_result,
            **identification_result,
            "final_verdict": detection_result['is_fraud'] or identification_result['suspicious_elements']
        }

    @staticmethod
    def detection_phase(transaction):
        # Your existing detection logic
        df = pd.DataFrame([transaction])
        df = pd.get_dummies(df, columns=['sender_country', 'receiver_country'])
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]
        
        scaled_features = scaler.transform(df)
        fraud_prob = model.predict_proba(scaled_features)[0][1]
        
        return {
            "fraud_probability": round(fraud_prob * 100, 2),
            "is_fraud": fraud_prob > 0.5
        }

    @staticmethod
    def identification_phase(transaction):
        suspicious_elements = []
        
        # Value analysis
        if transaction['amount'] > 25000:
            suspicious_elements.append("High value transaction")
        
        # Time window analysis
        if transaction['time_since_last_tx'] < 5 and transaction['amount'] > 5000:
            suspicious_elements.append("Rapid high-value transaction")
            
        # Network analysis
        if transaction['high_risk_sender'] and transaction['high_risk_receiver']:
            suspicious_elements.append("High-risk network")
            
        return {
            "suspicious_elements": suspicious_elements,
            "risk_score": len(suspicious_elements) * 25  # Simple scoring
        }

class SmartContract:
    @staticmethod
    def check_compliance(transaction):
        rules = [
            ("High value cross-border", 
             lambda t: t['is_cross_border'] and t['amount'] > 10000,
             "Requires additional verification"),
             
            ("High risk parties",
             lambda t: t['high_risk_sender'] or t['high_risk_receiver'],
             "Needs manual review"),
             
            ("Unusual frequency",
             lambda t: t['transaction_frequency'] > 15,
             "Possible structuring")
        ]
        
        violations = []
        for name, condition, action in rules:
            if condition(transaction):
                violations.append({
                    "rule": name,
                    "action": action
                })
                
        return violations

@app.route('/')
def index():
    stats = {
        'total_transactions': len(blockchain.chain) - 1,  # exclude genesis block
        'fraud_count': sum(1 for block in blockchain.chain if block.get('data', {}).get('is_fraud', False)),
        'last_block': blockchain.chain[-1] if blockchain.chain else None
    }
    return render_template('index.html', stats=stats)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Add these routes for user management
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        user = User.query.filter_by(username=username).first()
        
        if user and bcrypt.checkpw(password, user.password.encode('utf-8')):
            login_user(user)
            return redirect('/')
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        hashed = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())
        new_user = User(
            username=request.form['username'],
            password=hashed.decode('utf-8'),
            role='analyst'  # Default role
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        try:
            if not model or not scaler:
                raise Exception("Model not loaded. Please train the model first.")
            
            transaction = {
                "amount": float(request.form['amount']),
                "sender_balance": float(request.form['sender_balance']),
                "receiver_balance": float(request.form['receiver_balance']),
                "transaction_speed": int(request.form['transaction_speed']),
                "sender_country": request.form['sender_country'],
                "receiver_country": request.form['receiver_country'],
                "sender_id": request.form.get('sender_id', 'unknown'),
                "receiver_id": request.form.get('receiver_id', 'unknown'),
                "transaction_frequency": int(request.form.get('transaction_frequency', 1)),
                "time_since_last_tx": int(request.form.get('time_since_last_tx', 0)),
                "high_risk_sender": int(request.form.get('high_risk_sender', 0)),
                "high_risk_receiver": int(request.form.get('high_risk_receiver', 0))
            }
            
            # Calculate derived features
            transaction['is_cross_border'] = int(transaction['sender_country'] != transaction['receiver_country'])
            transaction['amount_to_balance_ratio'] = transaction['amount'] / transaction['sender_balance']
            
            # Create DataFrame
            df = pd.DataFrame([transaction])
            df = pd.get_dummies(df, columns=['sender_country', 'receiver_country'])
            
            # Ensure all features are present
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_names]
            
            # Scale and predict
            scaled_features = scaler.transform(df)
            fraud_prob = model.predict_proba(scaled_features)[0][1]
            is_fraud = fraud_prob > 0.5
            
            # Store in blockchain
            last_block = blockchain.get_last_block()
            block_data = {
                "transaction": transaction,
                "fraud_probability": round(fraud_prob * 100, 2),
                "is_fraud": bool(is_fraud),
                "analysis_time": str(datetime.now()),
                "risk_factors": get_risk_factors(transaction)
            }
            blockchain.create_block(last_block["hash"], block_data)
            
            return render_template('results.html', 
                                transaction=transaction,
                                fraud_prob=round(fraud_prob * 100, 2),
                                is_fraud=is_fraud,
                                block_index=len(blockchain.chain),
                                risk_factors=block_data["risk_factors"])
        
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return render_template('analyze.html')

def get_risk_factors(transaction):
    risk_factors = []
    
    # Amount-based risks
    if transaction['amount'] > 10000:
        risk_factors.append("High transaction amount (>10,000)")
    if transaction['amount_to_balance_ratio'] > 0.5:
        risk_factors.append(f"Large amount relative to balance ({transaction['amount_to_balance_ratio']:.2f})")
    
    # Speed and frequency
    if transaction['transaction_speed'] < 2:
        risk_factors.append("Very fast transaction")
    if transaction['transaction_frequency'] > 10:
        risk_factors.append(f"High transaction frequency ({transaction['transaction_frequency']})")
    
    # High-risk parties
    if transaction['high_risk_sender']:
        risk_factors.append("Sender is high-risk")
    if transaction['high_risk_receiver']:
        risk_factors.append("Receiver is high-risk")
    
    # Cross-border
    if transaction['is_cross_border']:
        risk_factors.append("Cross-border transaction")
    
    return risk_factors if risk_factors else ["No significant risk factors detected"]

@app.route('/transactions')
def view_transactions():
    return render_template('transactions.html', chain=blockchain.chain)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        data = request.get_json()
        # Implement similar to the analyze() function
        # Return JSON response
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            results = []

            for _, row in df.iterrows():
                # Convert row to dict and process like the analyze route
                transaction = row.to_dict()
                # Ensure required fields exist
                required_fields = ['sender_country', 'receiver_country', 'amount', 'sender_balance', 'high_risk_sender', 'high_risk_receiver']
                for field in required_fields:
                    if field not in transaction:
                        transaction[field] = 0  # or suitable default

                # Derived features for risk factors
                transaction['transaction_speed'] = transaction.get('transaction_speed', 0)
                transaction['transaction_frequency'] = transaction.get('transaction_frequency', 0)

                # Derived features
                transaction['is_cross_border'] = int(transaction['sender_country'] != transaction['receiver_country'])
                transaction['amount_to_balance_ratio'] = transaction['amount'] / (transaction['sender_balance'] + 1e-5)  # avoid divide by zero

                # Fix JSON serialization later
                transaction = {k: (v.item() if hasattr(v, 'item') else v) for k, v in transaction.items()}
                
                # Create DataFrame
                df_tx = pd.DataFrame([transaction])
                df_tx = pd.get_dummies(df_tx, columns=['sender_country', 'receiver_country'])
                df_tx['high_risk_sender'] = df_tx['high_risk_sender'].map({'Yes': 1, 'No': 0})
                df_tx['high_risk_receiver'] = df_tx['high_risk_receiver'].map({'Yes': 1, 'No': 0})

                
                # Ensure all features are present
                for col in feature_names:
                    if col not in df_tx.columns:
                        df_tx[col] = 0
                df_tx = df_tx[feature_names]
                
                # Scale and predict
                scaled_features = scaler.transform(df_tx)
                fraud_prob = model.predict_proba(scaled_features)[0][1]
                is_fraud = fraud_prob > 0.5
                
                # Store in blockchain
                last_block = blockchain.get_last_block()
                block_data = {
                    "transaction": transaction,
                    "fraud_probability": round(fraud_prob * 100, 2),
                    "is_fraud": bool(is_fraud),
                    "analysis_time": str(datetime.now()),
                    "risk_factors": get_risk_factors(transaction)
                }
                blockchain.create_block(last_block["hash"], block_data)
                
                results.append({
                    "transaction": transaction,
                    "fraud_probability": round(fraud_prob * 100, 2),
                    "is_fraud": is_fraud,
                    "block_index": len(blockchain.chain),
                    "risk_factors": block_data["risk_factors"]
                })
            
            return render_template('batch_results.html', results=results)
    return render_template('upload.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Fetch the blockchain data
    chain = blockchain.chain

    # Initialize counters
    risk_scores = []
    fraud_count = 0
    legit_count = 0

    for block in chain:
        data = block.get("data", {})
        risk_score = data.get("risk_score")
        is_fraud = data.get("is_fraud")

        if risk_score is not None:
            risk_scores.append(risk_score)
            if is_fraud:
                fraud_count += 1
            else:
                legit_count += 1

    # Precompute risk levels
    low_risk = len([r for r in risk_scores if r <= 30])
    med_risk = len([r for r in risk_scores if 30 < r <= 70])
    high_risk = len([r for r in risk_scores if r > 70])

    return render_template(
        "dashboard.html",
        fraud_count=fraud_count,
        legit_count=legit_count,
        low_risk=low_risk,
        med_risk=med_risk,
        high_risk=high_risk,
        chain=chain
    )


@app.route('/report/<int:block_id>')
@login_required
def generate_report(block_id):
    block = next((b for b in blockchain.chain if b['index'] == block_id), None)
    if not block:
        return render_template('error.html', error="Block not found")
    
    return render_template('report.html', block=block)



@app.context_processor
def inject_current_year():
    return {'current_year': datetime.utcnow().year}


if __name__ == '__main__':
    app.run(debug=True)