import json
import http.client
import urllib.parse
import sys

def test_ollama_tool_calling(model_name, host="localhost", port=11434):
    """
    Standalone test for Ollama tool calling with Gemma 4.
    No local imports allowed.
    """
    print(f"Testing tool calling for model: {model_name} on {host}:{port}...")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a given ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock ticker symbol (e.g., AAPL, TSLA)"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }
    ]

    messages = [
        {"role": "user", "content": "What is the current price of NVDA?"}
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    try:
        conn = http.client.HTTPConnection(host, port, timeout=30)
        headers = {"Content-Type": "application/json"}
        conn.request("POST", "/api/chat", json.dumps(payload), headers)
        
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        
        if response.status != 200:
            print(f"Error: Received status {response.status}")
            print(data)
            return False

        result = json.loads(data)
        message = result.get("message", {})
        tool_calls = message.get("tool_calls", [])

        if tool_calls:
            print("SUCCESS: Model correctly identified the tool call.")
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name")
                args = fn.get("arguments")
                print(f"  Tool: {name}")
                print(f"  Args: {args}")
            return True
        else:
            print("FAILURE: Model did not trigger the tool.")
            print("Response Content:")
            print(message.get("content", "(empty)"))
            return False

    except Exception as e:
        print(f"Exception during test: {e}")
        return False

if __name__ == "__main__":
    # Usage: python test_gemma4_tools.py <model_name> [host] [port]
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:26b"
    h = sys.argv[2] if len(sys.argv) > 2 else "localhost"
    p = int(sys.argv[3]) if len(sys.argv) > 3 else 11434
    
    success = test_ollama_tool_calling(model, h, p)
    sys.exit(0 if success else 1)
