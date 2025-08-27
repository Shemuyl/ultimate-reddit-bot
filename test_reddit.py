import os
import asyncio
import asyncpraw
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID     = os.getenv('REDDIT_CLIENT_ID')
CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
USERNAME      = os.getenv('REDDIT_USERNAME')
PASSWORD      = os.getenv('REDDIT_PASSWORD')
USER_AGENT    = os.getenv('REDDIT_USER_AGENT')

async def main():
    reddit = asyncpraw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        username=USERNAME,
        password=PASSWORD,
        user_agent=USER_AGENT
    )
    me = await reddit.user.me()
    print(f"Logged in as: {me.name}")

asyncio.run(main())
