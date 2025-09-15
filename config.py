import os
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o")
SIM_THRESHOLD_NEW_ATTR = float(os.getenv("SIM_THRESHOLD_NEW_ATTR", "0.78"))
TOPK_ATTR = int(os.getenv("TOPK_ATTR", "8"))
DATABASE_PATH = "data/data.db"
PROXY = f"http://um:qwertyui@51.250.120.68:4433"
