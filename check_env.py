import os
from dotenv import load_dotenv

load_dotenv()  # Reads your .env file

print("Discord token:", os.getenv("DISCORD_TOKEN"))
print("Channel ID:", os.getenv("DISCORD_CHANNEL_ID"))

