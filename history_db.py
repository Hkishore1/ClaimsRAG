# history_db.py
import sqlite3
import time
import uuid
from typing import List, Dict

DB_PATH = "agent_history.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
create table if not exists chat_history (
    id TEXT primary key,
    session_id TEXT,
    role TEXT,
    text TEXT,
    ts REAL
);
""")
conn.commit()


def add_turn(session_id: str, role: str, text: str):
    cursor.execute(
        "insert into chat_history (id, session_id, role, text, ts) values (?,?,?,?,?)",
        (uuid.uuid4().hex, session_id, role, text, time.time())
    )
    conn.commit()


def get_recent(session_id: str, limit: int = 20) -> List[Dict]:
    cursor.execute(
        "select role, text from chat_history where session_id=? order by ts desc limit ?",
        (session_id, limit)
    )
    rows = cursor.fetchall()
    return [{"role": r, "text": t} for r, t in rows]


def clear(session_id: str):
    cursor.execute("delete from chat_history where session_id=?", (session_id,))
    conn.commit()
