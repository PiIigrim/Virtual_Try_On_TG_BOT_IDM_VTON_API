import nest_asyncio
from pyngrok import ngrok
import os
from dotenv import load_dotenv
import threading

nest_asyncio.apply()

with open('.env', 'w') as f:
    f.write('TELEGRAM_BOT_TOKEN=7800799909:AAHlWp3ZOcS8y3-UvHqs63_6qjG7G76QU6s\n')
    f.write('API_BASE_URL=http://127.0.0.1:8000\n')
    f.write('HT_TOKEN=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\n')
    f.write('MODEL_NAME=yisol/IDM-VTON\n')
    f.write('JSON_DATA_URL=clothes.json\n')
    f.write('NGROK_AUTHTOKEN=2rLNSmD8FxiTpIsi3zo5jFYz1RB_z5GfLfb7fwrfZ8UGGz77')

load_dotenv()

token = os.getenv("NGROK_AUTHTOKEN")
ngrok.set_auth_token(token)

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)

def run_server():
    os.system('cd FASTAPI_server && python -m uvicorn server:app &')

server_thread = threading.Thread(target=run_server)

server_thread.start()

os.system('python E:\\other\\try_on\\Virtual_Try_On_TG_BOT_IDM_VTON_API\\Bot\\tg_bot.py')