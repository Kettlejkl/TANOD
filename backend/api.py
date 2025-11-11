from flask import Flask, jsonify, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
CORS(app)

# --- Database config ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tanod.db'  # or use MySQL/Postgres later
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# --- Example model ---
class Tanod(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    zone = db.Column(db.String(50))

# --- API routes ---
@app.route('/api/occupancy')
def get_occupancy():
    return jsonify({"Zone A": 5, "Zone B": 3})

@app.route('/api/alerts')
def get_alerts():
    return jsonify([
        {"id": 1, "type": "Overcrowding", "zone": "Zone A", "timestamp": "2025-08-08T10:00:00"}
    ])

@app.route('/api/video_feed')
def video_feed():
    def gen():
        with open('static/placeholder.jpg', 'rb') as f:
            frame = f.read()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
