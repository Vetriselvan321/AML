from app import app, db
from app.models import User
import bcrypt

with app.app_context():
    db.create_all()
    
    if not User.query.filter_by(username='admin').first():
        hashed = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
        admin = User(username='admin', password=hashed.decode('utf-8'), role='admin')
        db.session.add(admin)
        db.session.commit()
        print("Admin user created.")
    else:
        print("Admin user already exists.")
