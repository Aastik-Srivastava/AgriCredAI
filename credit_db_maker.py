
import sqlite3
import hashlib
from datetime import datetime


DB_PATH = "carbon_credits.db"
CREDIT_PRICE_USD = 6.5  # Example voluntary market price per tCO₂e
USD_TO_INR = 83.0
CAR_EQUIV_TON = 4.6  # 1 car emits 4.6 tCO₂e/year
TREE_EQUIV_TON = 0.022  # 1 tree absorbs 0.022 tCO₂e/year


# --- Initialize Database ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS credits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    farm_id TEXT,
                    location TEXT,
                    date TEXT,
                    verification_status TEXT,
                    credits REAL,
                    hash TEXT,
                    prev_hash TEXT
                )''')
    conn.commit()
    conn.close()

init_db()



# --- Helper: Store credit transaction in blockchain-style ledger ---
def store_credit_transaction(farm_id, location, verification_status, credits):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT hash FROM credits ORDER BY id DESC LIMIT 1")
    prev_hash = cur.fetchone()
    prev_hash = prev_hash[0] if prev_hash else "0"

    tx_data = f"{farm_id}{location}{datetime.utcnow().isoformat()}{verification_status}{credits}{prev_hash}"
    tx_hash = hashlib.sha256(tx_data.encode()).hexdigest()

    cur.execute("INSERT INTO credits (farm_id, location, date, verification_status, credits, hash, prev_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (farm_id, location, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), verification_status, credits, tx_hash, prev_hash))
    conn.commit()
    conn.close()
