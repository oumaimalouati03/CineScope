"""Small SQLite store for demo profiles, personal ratings, favorites and viewing history."""
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parents[1] / 'data' / 'cinescope_user.db'

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript('''
      CREATE TABLE IF NOT EXISTS profiles (username TEXT PRIMARY KEY, created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS ratings (username TEXT NOT NULL, movie_id INTEGER NOT NULL, rating REAL NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(username,movie_id));
      CREATE TABLE IF NOT EXISTS favorites (username TEXT NOT NULL, movie_id INTEGER NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(username,movie_id));
      CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, movie_id INTEGER NOT NULL, action TEXT NOT NULL, created_at TEXT NOT NULL);
    ''')
    con.commit()
    return con

def ensure_profile(con, username):
    con.execute('INSERT OR IGNORE INTO profiles VALUES (?,?)', (username, datetime.now().isoformat(timespec='seconds'))); con.commit()

def save_rating(con, username, movie_id, rating):
    now = datetime.now().isoformat(timespec='seconds')
    con.execute('INSERT INTO ratings VALUES (?,?,?,?) ON CONFLICT(username,movie_id) DO UPDATE SET rating=excluded.rating, updated_at=excluded.updated_at', (username, int(movie_id), float(rating), now))
    con.execute('INSERT INTO history(username,movie_id,action,created_at) VALUES (?,?,?,?)', (username,int(movie_id),f'Note : {rating:g}/5',now)); con.commit()

def toggle_favorite(con, username, movie_id):
    cur = con.execute('SELECT 1 FROM favorites WHERE username=? AND movie_id=?', (username,int(movie_id))).fetchone()
    now = datetime.now().isoformat(timespec='seconds')
    if cur:
        con.execute('DELETE FROM favorites WHERE username=? AND movie_id=?', (username,int(movie_id))); action='Retiré des favoris'
    else:
        con.execute('INSERT INTO favorites VALUES (?,?,?)', (username,int(movie_id),now)); action='Ajouté aux favoris'
    con.execute('INSERT INTO history(username,movie_id,action,created_at) VALUES (?,?,?,?)', (username,int(movie_id),action,now)); con.commit()
    return not bool(cur)

def log_view(con, username, movie_id):
    con.execute('INSERT INTO history(username,movie_id,action,created_at) VALUES (?,?,?,?)', (username,int(movie_id),'Consultation',datetime.now().isoformat(timespec='seconds'))); con.commit()

def get_ratings(con, username): return con.execute('SELECT movie_id, rating, updated_at FROM ratings WHERE username=? ORDER BY updated_at DESC',(username,)).fetchall()
def get_favorites(con, username): return con.execute('SELECT movie_id, created_at FROM favorites WHERE username=? ORDER BY created_at DESC',(username,)).fetchall()
def get_history(con, username, limit=100): return con.execute('SELECT movie_id, action, created_at FROM history WHERE username=? ORDER BY id DESC LIMIT ?',(username,limit)).fetchall()
