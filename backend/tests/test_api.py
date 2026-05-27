import pytest
from app import create_app, db
from app.models import Zone, Alert, AlertType

@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

def test_occupancy_endpoint(client):
    # Create test zone
    zone = Zone(name='Test Zone', max_capacity=10)
    db.session.add(zone)
    db.session.commit()
    
    response = client.get('/api/occupancy/current')
    assert response.status_code == 200
    data = response.get_json()
    assert 'Test Zone' in data