import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import app, get_genai_client, init_db, get_db

def test_gemini_chatbot():
    print("=== Testing AyushBot with Gemini API Key ===")
    
    # Check Gemini Client
    client = get_genai_client()
    print("1. Gemini Client Initialized:", client is not None)
    assert client is not None, "Failed to initialize Gemini Client"
    
    app.config['TESTING'] = True
    test_client = app.test_client()
    
    with test_client.session_transaction() as sess:
        sess['role'] = 'patient'
        sess['user_id'] = 1
        sess['user_name'] = 'Priya Nair'
        
    # Test GET /patient/chatbot
    resp = test_client.get('/patient/chatbot')
    print("2. GET /patient/chatbot Status Code:", resp.status_code)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert b"AyushBot" in resp.data, "AyushBot title not found in HTML"
    assert b"Priya Nair" in resp.data, "Patient name not found in HTML"
    print("   Page rendered with AyushBot and patient profile successfully.")

    # Test POST /api/patient/chatbot with health query
    print("3. Testing POST /api/patient/chatbot with query...")
    payload = {
        'message': 'What are 2 good Ayurvedic home remedies for occasional digestion issues and bloating?',
        'history': [],
        'lang': 'en'
    }
    api_resp = test_client.post('/api/patient/chatbot', json=payload)
    print("   POST Status Code:", api_resp.status_code)
    assert api_resp.status_code == 200, f"Expected 200, got {api_resp.status_code}: {api_resp.data}"
    
    data = api_resp.get_json()
    print("   Response success:", data.get('success'))
    print("   Model used:", data.get('model'))
    print("   Reply snippet:", data.get('reply')[:200] if data.get('reply') else 'None')
    assert data.get('success') is True, f"Chatbot failed: {data}"
    assert len(data.get('reply', '')) > 20, "Reply too short"
    print("[SUCCESS] AyushBot Gemini Chatbot integration verified successfully!")

if __name__ == '__main__':
    test_gemini_chatbot()
