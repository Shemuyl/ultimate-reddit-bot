import asyncio
import os
from bot import discord_client, mk_reddit

async def test_discord():
    async with discord_client:  # Properly logs in and out
        channel_id = int(os.getenv('DISCORD_CHANNEL_ID'))
        channel = discord_client.get_channel(channel_id)
        if channel:
            await channel.send("✅ Test message from bot!")
            print("✅ Discord message sent successfully!")
        else:
            print("❌ Discord channel not found. Check DISCORD_CHANNEL_ID.")

async def test_reddit():
    reddit = mk_reddit()
    me = await reddit.user.me()
    print(f"✅ Logged in to Reddit as: {me.name}")

async def main():
    await test_discord()
    await test_reddit()

if __name__ == "__main__":
    asyncio.run(main())
