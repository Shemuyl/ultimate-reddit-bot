#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Reddit Bug-Fixer Bot v18.2 (Discord removed)
- Fully Reddit-only
- All features intact: AI, code sandboxing, lead scoring, follow-ups, resilient loops
"""

import os
import asyncio
import random
import json
import sqlite3
import logging
import re
import subprocess
import tempfile
import signal
import contextlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
import asyncpraw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from logging.handlers import RotatingFileHandler

# ---------- Environment ----------
load_dotenv()

def _parse_list(s: str, default: str = ""):
    s = (s or default).strip()
    if not s:
        return []
    return [x.strip() for x in re.split(r"[,\s]+", s) if x.strip()]

CLIENT_ID     = (os.getenv('REDDIT_CLIENT_ID') or "").strip()
CLIENT_SECRET = (os.getenv('REDDIT_CLIENT_SECRET') or "").strip()
USERNAME      = (os.getenv('REDDIT_USERNAME') or "").strip()
PASSWORD      = (os.getenv('REDDIT_PASSWORD') or "").strip()
USER_AGENT    = (os.getenv('REDDIT_USER_AGENT') or f'Ultimate Bug-Fixer v18 by u/{USERNAME}').strip()

TARGET_SUBS      = _parse_list(os.getenv('TARGET_SUBREDDITS'), 'slavelabour,forhire')
POST_COOLDOWN_HR = float(os.getenv('POST_COOLDOWN_HOURS') or '24')

KEYWORDS         = _parse_list(os.getenv('KEYWORDS'), 'bug,fix,error,issue,broken,help,debug,css,python,javascript,js')
_ad_default_json = '[{"title":"Fast bug fixes ($20–$100)","body":"We fix your bugs fast and professionally."}]'
try:
    AD_MESSAGES = json.loads(os.getenv('AD_MESSAGES') or _ad_default_json)
    if not isinstance(AD_MESSAGES, list) or not AD_MESSAGES:
        AD_MESSAGES = json.loads(_ad_default_json)
except Exception:
    AD_MESSAGES = json.loads(_ad_default_json)

FOLLOWUP_HOURS   = float(os.getenv('FOLLOWUP_HOURS') or '12')

DB_PATH = Path("crm.db")
LOG_FILE = Path("bot.log")

# ---------- Logging ----------
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
console = logging.StreamHandler()
logging.basicConfig(level=logging.INFO, handlers=[handler, console], format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- Optional AI Reply ----------
try:
    import openai
    OPENAI_KEY = (os.getenv('OPENAI_API_KEY') or "").strip()
    openai.api_key = OPENAI_KEY
except Exception:
    OPENAI_KEY = None

# ---------- Database ----------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS leads (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    author TEXT,
                    subject TEXT,
                    preview TEXT,
                    permalink TEXT,
                    score INTEGER,
                    status TEXT
                 )""")
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    lead_id INTEGER,
                    message TEXT,
                    reply TEXT,
                    timestamp TEXT
                 )""")
    c.execute("""CREATE TABLE IF NOT EXISTS followups (
                    id INTEGER PRIMARY KEY,
                    lead_id INTEGER,
                    reddit_id TEXT UNIQUE,
                    followup_time TEXT
                 )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_leads_author ON leads(author)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_messages_lead ON messages(lead_id)")
    conn.commit()
    return conn

# ---------- Reddit ----------
def mk_reddit():
    missing = [k for k,v in [
        ('REDDIT_CLIENT_ID',CLIENT_ID),
        ('REDDIT_CLIENT_SECRET',CLIENT_SECRET),
        ('REDDIT_USERNAME',USERNAME),
        ('REDDIT_PASSWORD',PASSWORD),
        ('REDDIT_USER_AGENT',USER_AGENT)
    ] if not v]
    if missing:
        raise SystemExit(f"Missing Reddit credentials: {missing}")
    return asyncpraw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
                            username=USERNAME, password=PASSWORD, user_agent=USER_AGENT)

# ---------- Sentiment & Lead Scoring ----------
analyzer = SentimentIntensityAnalyzer()
def analyze_sentiment(text):
    try:
        return analyzer.polarity_scores(text)['compound']
    except Exception:
        return 0.0

def dynamic_score_lead(text: str) -> int:
    base = 0
    text_l = (text or "").lower()
    for kw in KEYWORDS:
        if kw and kw in text_l:
            base += 10
    techs = ['python','javascript','css','react','node','html','django','flask','vue','typescript']
    for tech in techs:
        if tech in text_l:
            base += 5
    urgency = ['urgent','asap','deadline','immediately','critical','fix now']
    if any(u in text_l for u in urgency):
        base += 20
    sentiment = analyze_sentiment(text_l)
    if sentiment < -0.2:
        base += 15
    if "```" in text_l:
        base += 25
    return min(base, 100)

# ---------- Code Detection & Sandbox ----------
CODE_BLOCK_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
def extract_code_blocks(text):
    return [(m.group(1) or 'text', m.group(2)) for m in CODE_BLOCK_RE.finditer(text or "")]

def run_sandboxed_code(lang: str, code: str) -> str:
    try:
        if lang.lower() == 'python':
            result = subprocess.run(['python3','-c',code], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return result.stderr
        elif lang.lower() in ['js','javascript']:
            result = subprocess.run(['node','-e',code], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return result.stderr
        elif lang.lower() in ['html','css']:
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{lang.lower()}', delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                if lang.lower() == 'html':
                    result = subprocess.run(['tidy', '-qe', tmp_path], capture_output=True, text=True)
                    if result.returncode != 0:
                        return result.stderr
                else:
                    result = subprocess.run(['csslint', tmp_path], capture_output=True, text=True)
                    if "Error" in result.stdout or "Error" in result.stderr:
                        return result.stdout + result.stderr
            finally:
                os.remove(tmp_path)
        return None
    except Exception as e:
        return str(e)

# ---------- Async AI Rate-Limiter ----------
AI_SEMAPHORE = asyncio.Semaphore(2)

async def iterative_ai_fix(lang, code, max_iter=4):
    if not OPENAI_KEY:
        return "AI disabled (no OPENAI_API_KEY)."
    last_fix = code
    async with AI_SEMAPHORE:
        for i in range(max_iter):
            prompt = f"You are a professional {lang} developer. Here is the code:\n{last_fix}\nProvide a fixed version in triple backticks."
            try:
                resp = await asyncio.to_thread(lambda: openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=700
                ))
                fix_text = resp.choices[0].message.content
                extracted = extract_code_blocks(fix_text)
                if extracted and extracted[0][1].strip():
                    last_fix = extracted[0][1]
                    sandbox_result = run_sandboxed_code(lang, last_fix)
                    if not sandbox_result:
                        logging.info(f"{lang} sandbox passed at iteration {i+1}")
                        return f"{fix_text}\n✅ Sandbox passed"
                    else:
                        logging.warning(f"{lang} sandbox failed at iteration {i+1}: {sandbox_result}")
                        fix_text += f"\n⚠️ Sandbox failed: {sandbox_result}\nRetrying iteration {i+1}"
                else:
                    return fix_text
            except Exception as e:
                logging.error(f"AI iterative fix error ({lang}): {e}")
                return f"AI fix error: {e}"
    return f"{last_fix}\n⚠️ Maximum iterations reached, may still contain errors"

async def generate_reply(text: str, history: list = []) -> str:
    if not OPENAI_KEY:
        return AD_MESSAGES[0]['body']
    code_blocks = extract_code_blocks(text)
    if code_blocks:
        fixes = []
        for lang, code in code_blocks:
            fixes.append(await iterative_ai_fix(lang, code))
        return "\n\n".join(fixes)
    hist_str = "\n".join([f"User: {m['message']}\nBot: {m['reply']}" for m,r in history[-10:]])
    prompt = f"Conversation history:\n{hist_str}\n\nNew message:\n{text}\nProvide a concise, professional reply:"
    try:
        async with AI_SEMAPHORE:
            resp = await asyncio.to_thread(lambda: openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=400
            ))
        return resp.choices[0].message.content
    except Exception as e:
        logging.error(f"AI error: {e}")
        return AD_MESSAGES[0]['body']

# ---------- Safe call ----------
async def safe_call(callable_or_coro, *args, **kwargs):
    backoff = 5
    while True:
        try:
            if asyncio.iscoroutine(callable_or_coro):
                return await callable_or_coro
            if asyncio.iscoroutinefunction(callable_or_coro):
                return await callable_or_coro(*args, **kwargs)
            return await asyncio.to_thread(callable_or_coro, *args, **kwargs)
        except Exception as e:
            logging.warning(f"API error: {e} – backoff {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff*2, 600)

# ---------- Resilient wrapper ----------
async def resilient(task_func, *args):
    while True:
        try:
            await task_func(*args)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.error(f"Task {task_func.__name__} failed: {e}, restarting in 10s")
            await asyncio.sleep(10)

# ---------- Ads ----------
async def post_ads(reddit):
    posted_path = Path('post_log.json')
    post_log = json.loads(posted_path.read_text(encoding='utf-8')) if posted_path.exists() else {}
    while True:
        now = datetime.now(timezone.utc)
        for sub in TARGET_SUBS:
            try:
                last_ts = post_log.get(sub)
                if last_ts and now < datetime.fromisoformat(last_ts) + timedelta(hours=POST_COOLDOWN_HR):
                    continue
                subreddit = await reddit.subreddit(sub)
                ad = max(AD_MESSAGES, key=lambda x: x.get('engagement', 0))
                submission = await subreddit.submit(title=ad['title'], selftext=ad['body'], send_replies=True)
                post_log[sub] = now.isoformat()
                tmp_path = posted_path.with_suffix('.tmp')
                tmp_path.write_text(json.dumps(post_log, indent=2))
                tmp_path.replace(posted_path)
                logging.info(f"📝 Posted ad to r/{sub}: {submission.permalink}")
                await asyncio.sleep(random.randint(15,30))
            except Exception as e:
                logging.error(f"Error posting ad to r/{sub}: {e}")
                await asyncio.sleep(20)
        await asyncio.sleep(600)

# ---------- Inbox & comments ----------
async def run_inbox(reddit, conn):
    c = conn.cursor()
    seen_path = Path('processed_pm_ids.txt')
    seen = set(seen_path.read_text(encoding="utf-8").splitlines()) if seen_path.exists() else set()
    followup_path = Path('followup_queue.json')
    try:
        followups = json.loads(followup_path.read_text(encoding="utf-8")) if followup_path.exists() else {}
    except Exception:
        followups = {}

    while True:
        try:
            async for item in reddit.inbox.unread(limit=None):
                if item.id in seen:
                    await item.mark_read()
                    continue
                author = getattr(item.author, 'name', '(unknown)')
                subj = getattr(item, 'subject', '(no subject)')
                body_preview = (getattr(item, 'body', '') or '')[:400]
                permalink = f"https://reddit.com{getattr(item, 'context', '')}" if hasattr(item,'context') else ''
                score = dynamic_score_lead(body_preview)
                with conn:
                    conn.execute("INSERT INTO leads (timestamp, author, subject, preview, permalink, score, status) VALUES (?,?,?,?,?,?,?)",
                                 (datetime.now(timezone.utc).isoformat(), author, subj, body_preview, permalink, score, 'new'))
                    lead_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                c.execute("SELECT message, reply FROM messages WHERE lead_id=? ORDER BY id", (lead_id,))
                history = [{"message":m,"reply":r} for m,r in c.fetchall()]
                reply_text = await generate_reply(body_preview, history)
                await item.reply(reply_text)
                await item.mark_read()
                seen.add(item.id)
                tmp_seen = seen_path.with_suffix('.tmp')
                tmp_seen.write_text("\n".join(seen), encoding="utf-8")
                tmp_seen.replace(seen_path)
                with conn:
                    conn.execute("INSERT INTO messages (lead_id, message, reply, timestamp) VALUES (?,?,?,?)",
                                 (lead_id, body_preview, reply_text, datetime.now(timezone.utc).isoformat()))
                if score >= 40:
                    logging.info(f"🔥 High-Priority Lead ({score}): u/{author} | {subj}\n{body_preview}")
                else:
                    logging.info(f"📩 Lead ({score}): u/{author} | {subj}")

            now = datetime.now(timezone.utc)
            for fid, ts in list(followups.items()):
                if now >= datetime.fromisoformat(ts):
                    try:
                        msg = await reddit.inbox.message(fid)
                        followup_text = f"Just checking in regarding your bug request. {AD_MESSAGES[0]['body']}"
                        await msg.reply(followup_text)
                        logging.info(f"⏰ Followed up on u/{msg.author.name if msg.author else '(unknown)'}")
                        followups.pop(fid)
                    except Exception as e:
                        logging.error(f"Followup error for {fid}: {e}")

            tmp_follow = followup_path.with_suffix('.tmp')
            tmp_follow.write_text(json.dumps(followups, indent=2), encoding="utf-8")
            tmp_follow.replace(followup_path)
            await asyncio.sleep(10)
        except Exception as e:
            logging.error(f"Inbox loop error: {e}")
            await asyncio.sleep(10)

# ---------- Watch subreddit ----------
async def watch_subreddit(reddit, conn, sub):
    replied_path = Path(f'replied_{sub}.txt')
    replied = set(replied_path.read_text(encoding="utf-8").splitlines()) if replied_path.exists() else set()
    c = conn.cursor()
    subreddit = await reddit.subreddit(sub)
    while True:
        try:
            async for comment in subreddit.stream.comments(skip_existing=True):
                if comment.author and comment.author.name == USERNAME:
                    continue
                if comment.id in replied:
                    continue
                text = comment.body or ''
                score = dynamic_score_lead(text)
                if any(k in text.lower() for k in KEYWORDS):
                    c.execute("SELECT m.message, m.reply FROM messages m JOIN leads l ON m.lead_id=l.id WHERE l.author=? ORDER BY m.id DESC LIMIT 5",
                              (comment.author.name if comment.author else '',))
                    history=[{"message":m,"reply":r} for m,r in c.fetchall()]
                    reply_text = await generate_reply(text, history)
                    await comment.reply(reply_text)
                    replied.add(comment.id)
                    tmp_path = replied_path.with_suffix('.tmp')
                    tmp_path.write_text("\n".join(replied), encoding="utf-8")
                    tmp_path.replace(replied_path)
                    with conn:
                        conn.execute("INSERT INTO messages (lead_id, message, reply, timestamp) VALUES ((SELECT id FROM leads WHERE author=? ORDER BY id DESC LIMIT 1),?,?,?)",
                                     (comment.author.name if comment.author else '', text, reply_text, datetime.now(timezone.utc).isoformat()))
                    if score >= 40:
                        logging.info(f"🔥 High-Priority Comment ({score}) by u/{comment.author.name if comment.author else ''} in r/{sub}")
                    else:
                        logging.info(f"💬 Comment ({score}) by u/{comment.author.name if comment.author else ''} in r/{sub}")
        except Exception as e:
            logging.error(f"Watch {sub} error: {e}")
            await asyncio.sleep(10)

async def watch_all_subs(reddit, conn):
    if not TARGET_SUBS:
        logging.warning("No TARGET_SUBREDDITS configured; watcher will idle.")
        await asyncio.Future()
    await asyncio.gather(*[resilient(watch_subreddit, reddit, conn, s) for s in TARGET_SUBS])

# ---------- Main ----------
async def main():
    reddit = mk_reddit()
    conn = init_db()

    try:
        await asyncio.gather(
            resilient(post_ads, reddit),
            resilient(run_inbox, reddit, conn),
            watch_all_subs(reddit, conn)
        )
    except asyncio.CancelledError:
        logging.info("Main cancelled — shutting down")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            pass
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        pending = asyncio.all_tasks(loop=loop)
        for t in pending:
            t.cancel()
        with contextlib.suppress(Exception):
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
