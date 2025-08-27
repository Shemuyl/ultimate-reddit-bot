import asyncio
from bot import notify_discord, discord_client
import os

async def main():
    async with discord_client:  # This logs in and manages the session automatically
        await notify_discord("✅ Test message from bot!")
        await asyncio.sleep(2)  # wait a moment to ensure message is sent

asyncio.run(main())
