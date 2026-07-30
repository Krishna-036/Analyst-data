import os
import json
import io
import contextlib
import requests
import pandas as pd
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
import openai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HOST_URL = os.getenv("HOST_URL")  # e.g., https://my-bot.onrender.com

# Setup public directory for logs
os.makedirs("public", exist_ok=True)
app.mount("/public", StaticFiles(directory="public"), name="public")

chat_histories = {}
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def execute_python(code: str) -> str:
    """Executes python code dynamically. Allows the LLM to use pandas to read/analyze data."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            # Pass pandas, json, and requests into the exec environment
            exec(code, {"pd": pd, "json": json, "requests": requests})
        except Exception as e:
            print(f"Error: {e}")
    return f.getvalue()

def process_message(chat_id: int, text: str):
    # 1. Manage Multi-turn context
    if chat_id not in chat_histories:
        chat_histories[chat_id] = [
            {"role": "system", "content": "You are a Python data analyst agent. You will receive data queries. Use the `execute_python` tool to write and run pandas code to download and analyze the datasets. Print your intermediate findings so you can see them. Finally, output EXACTLY the requested JSON format."}
        ]
    
    chat_histories[chat_id].append({"role": "user", "content": text})
    
    # 2. Heuristic: Wait for the final instruction before running analysis
    # If the user is just passing multi-turn context and not asking for the final JSON, skip replying.
    if "{" not in text and "json" not in text.lower():
        return

    # 3. Agent Loop with Tool Calling
    messages = chat_histories[chat_id].copy()
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code to analyze data. Use print() to output results to yourself.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "The python code to run."}},
                "required": ["code"]
            }
        }
    }]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools
        )
        msg = response.choices[0].message
        messages.append(msg)
        
        # If the LLM wants to run Python, execute it and feed the result back
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "execute_python":
                    args = json.loads(tool_call.function.arguments)
                    result = execute_python(args["code"])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
        else:
            final_text = msg.content.strip()
            break

    # 4. Extract and shape the exact JSON required
    if final_text.startswith("```json"): final_text = final_text[7:]
    if final_text.startswith("```"): final_text = final_text[3:]
    if final_text.endswith("```"): final_text = final_text[:-3]
    final_text = final_text.strip()
    
    try:
        ans = json.loads(final_text)
        # Strip out the LLM's hallucinated log_url or nested answer key if present
        if "log_url" in ans: del ans["log_url"]
        if "answer" in ans and len(ans) == 1: ans = ans["answer"]
    except Exception:
        ans = {"raw": final_text}

    final_reply = {
        "answer": ans,
        "log_url": f"{HOST_URL}/public/run.jsonl"
    }

    # 5. Log as JSONL
    with open("public/run.jsonl", "a") as f:
        f.write(json.dumps({"question": text, "reply": final_reply}) + "\n")

    # 6. Reply exactly with the JSON
    requests.post(
        f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_TOKEN}/sendMessage",
        json={"chat_id": chat_id, "text": json.dumps(final_reply)}
    )

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives updates from Telegram."""
    data = await request.json()
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"]["text"]
        # Run agent in background so Telegram doesn't timeout the webhook
        background_tasks.add_task(process_message, chat_id, text)
    return {"status": "ok"}

@app.on_event("startup")
def on_startup():
    """Automatically registers the webhook on deployment."""
    requests.get(f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_TOKEN}/setWebhook?url={HOST_URL}/webhook")
