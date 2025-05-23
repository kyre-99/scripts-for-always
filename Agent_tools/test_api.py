import requests

BASE_URL = "http://127.0.0.1:8001"


def test_rag():
    data = {
        "query": "test",
    }
    response = requests.post(f"{BASE_URL}/rag", json=data)
    print("注册响应:", response.status_code, response.json())
    return response.json()
if __name__ == '__main__':
    test_rag()