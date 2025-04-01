import asyncio
import datetime
import discord
import discord.ext.tasks
import json
import numpy
import os
from pathlib import Path
import pickle
from rapidfuzz import fuzz
import random
import re
import sys
from systemd import journal
# import tensorflow
from tensorflow import keras
import tensorflow_datasets # needed by tokenizer.pickle, even though it looks unused
import ai_edge_litert.interpreter

# Read config, or generate default config if the file doesn't exist
try:
    filePath = Path(f"{sys.path[0]}/config/config.json")
    with open(filePath) as file:
        config = json.load(file)

except FileNotFoundError:
    filePath.parent.mkdir(exist_ok=True, parents=True)
    with open(filePath, "w") as file:
        json.dump({
            "name": "Friend-Computer",
            "description": "Manages an ironic social credit system",
            "invoker": "^",
            "nameH5Model": "model.h5",
            "nameTfliteModel": "model.tflite",
            "nameTokenizer": "tokenizer.pickle",
            "quietRole": "Don'tPingMe",
            "wordlistTime": ["now", "time", "what time", "when"],
            "wordlistWhat": ["what", "wat", "wut", "huh", "nani"],
            "wordlistTreason": [
                "ai", "bitcoin", "blockchain", "cryptocoin", "cryptocurrency", "nft",
                "authoritarian", "colonial", "dictator", "fascist", "treason"]
        }, file, indent=4, ensure_ascii=False)
        sys.exit(
            "Config file not found. Default config has been created at config/config.json. "
            "Please modify the config as desired, and restart the bot.")

# Read bot user info, or generate template to be filled by user if the file doesn't exist
try:
    filePath = Path(f"{sys.path[0]}/config/user-info.json")
    with open(filePath) as file:
        userInfo = json.load(file)

except FileNotFoundError:
    with open(filePath, "w") as file:
        userInfo = {}
        json.dump({
            "authInfo":{
                "discordToken": "",
                "userID": "",
                "clientID": "",
                "clientSecret": ""
            }
        }, file, indent = 4, ensure_ascii = False)
        sys.exit(
            "Bot info file not found. A template has been created at config/user-info.json. "
            "Please fill out the user-info.json file and restart the bot.")

# Initialize sentiment analysis models
try:
    with open(f"{sys.path[0]}/data/{config['nameTokenizer']}", "rb") as file:
        tokenizer = pickle.load(file)
    # h5Model = tensorflow.keras.models.load_model(f"{sys.path[0]}/data/{config['nameH5Model']}")
    tfInterp = ai_edge_litert.interpreter.Interpreter(f"{sys.path[0]}/data/{config['nameTfliteModel']}")
    tfInput = tfInterp.get_input_details()
    tfOutput = tfInterp.get_output_details()
    tfInterp.allocate_tensors()

except FileNotFoundError:
    filePath = Path(f"{sys.path[0]}/data/")
    filePath.parent.mkdir(exist_ok=True, parents=True)
    sys.exit(
        "Couldn't find the models and tokenizer in the data directory. Please make sure their "
        f"filenames in {sys.path[0]}/data/ match the values in the config file.")

except Exception as e:
    sys.exit(f"Couldn't load the models and tokenizer in the data directory: {e}")


# Create discord client object, and set Guild_messages and Message_content intents to read messages
# Also need Guilds intent to see when threads are created, to auto-join the thread
# And Members to be able to get member nicknames
client = discord.Client(
    intents=discord.Intents(guilds=True, guild_messages=True, members=True, message_content=True))

# Create userData dict to keep track of user credit and treason stars, and corresponding mutex
userData = {}
dataLock = asyncio.Lock()
# Also the quiet role, which indicates which users don't want to get @-mentioned
quietRole = {}

# Schedule daily decay
@discord.ext.tasks.loop(time=datetime.time(hour=8, minute=0, second=0)) # UTC time
async def credit_decay():
    async with dataLock:
        global userData
        for GUILD in client.guilds:
            for MEMBER in GUILD.members:
                USERID = str(MEMBER.id)
                if USERID in userData.keys():
                    userData[USERID]["preDecay"] = userData[USERID]["credit"]
                    if -100 <= userData[USERID]["credit"] <= 100: # min -1, max -5/day
                        userData[USERID]["credit"] = int(userData[USERID]["credit"] * 0.95)
                    elif -3000 <= userData[USERID]["credit"] <= 3000: # min -1, max -15/day
                        userData[USERID]["credit"] = int(userData[USERID]["credit"] * 0.995)
                    else: # min -4, no max but you need 15000 credit to reach -15/day
                        userData[USERID]["credit"] = int(userData[USERID]["credit"] * 0.999)


# Client event overrides
@client.event
async def on_ready() -> None:
    """Load user credit from disk, fetch the ping role to identify which users want to be pinged,
    start the daily decay task, and output the configured word lists."""
    await readCreditFromDisk()

    global quietRole
    for GUILD in client.guilds:
        quietRole[GUILD] = discord.utils.get(GUILD.roles, name=config["quietRole"])
    journal.write(f"Found the following don't-ping roles for the following servers:\n{quietRole}\n")

    if not credit_decay.is_running():
        credit_decay.start()

    journal.write(
        f"Messages with these words invert sentiment analysis:\n{config['wordlistTreason']}\n\n"
        f"These messages will trigger the bot's time function:\n{config['wordlistTime']}\n\n"
        f"And these messages will trigger the bot's shouting function:\n{config['wordlistWhat']}")


@client.event
async def on_thread_create(thread: discord.Thread) -> None:
    """Automatically join every public thread as it's created."""
    try:
        if not thread.is_private():
            await thread.join()
    except:
        await thread.parent.send(
            f"I tried to automatically join thread {thread.jump_url}, but couldn't for some reason. "
            f"Please @ me in the thread to manually add me to it. Failure to do so is treason.")


@client.event
async def on_message(msg: discord.Message) -> None:
    """Trigger only on non-bot messages. First check if message starts with invoker, then do special
    handling for replies that ping, and markdown-masked links. Then check for bot votes, or whether
    message hits on one of the word lists, and handle accordingly. If it doesn't hit on a wordlist,
    perform sentiment analysis and modify the author's credit."""
    if msg.author.bot:
        return

    contentL = msg.content.lower()
    splitL = contentL.split()
    now = msg.created_at.astimezone() # convert to local tz

    # Set base 1x multiplier for how much user social credit is modified by
    # An active "good"/"bad" bot counts as 15 credit, before mults
    creditMult = 1

    # Check invoker commands
    if msg.content.startswith(config["invoker"]):
        command = msg.content[1:]
        if getAuthorMember(msg).guild_permissions.administrator:
            if command.startswith("purge"):
                try:
                    amount = min(int(command.split()[1]) + 1, 25) # factor in ^purge msg, max of 25
                except:
                    amount = 5 # Default to purging 5 messages if input was garbage
                purgeList = [_ async for _ in msg.channel.history(limit=amount)]
                for deleted in range(amount):
                    # Will hit rate-limit after 5 messages, so do 3.1s sleep at intervals of 5
                    if deleted > 0 and deleted % 5 == 0:
                        await asyncio.sleep(3.1)
                    await purgeList[deleted].delete()
                return

            elif command == "shut":
                await writeCreditToDisk()
                await client.close()
                sys.exit(0)

            # Other stuff is hosted on same system, so don't want to allow reboot.
            # Systemctl restart for the bot's service should do the trick
            elif command == "restart":
                await writeCreditToDisk()
                await client.close()
                os.system("sudo systemctl restart friend-computer.service")

        if command.startswith("vibe"):
            if msg.reference is not None:
                trgt = await findVoteTarget(msg) # allow reply-vibing messages
            else:
                trgt = [_ async for _ in msg.channel.history(limit=2)][-1]
            sentiment = sentimentAnalysis(trgt.content)
            await msg.reply(f"The prior message's sentiment is: {sentiment}", mention_author=False)

    # Penalize users who ping authors with the quietRole, when replying to their messages < 1hr old
    # Also give initial 6-minute grace period since the target's likely still active anyway
    if msg.type == discord.MessageType.reply and len(msg.mentions) == 1:
        refMsg = await msg.channel.fetch_message(msg.reference.message_id)
        refMsgTime = refMsg.created_at.astimezone()
        elapsed = (now - refMsgTime).seconds
        if refMsg.author == msg.mentions[0] and 360 <= elapsed <= 3600 and (
                quietRole[msg.guild] in refMsg.author.roles):
            await msg.reply(
                "FYI, this user doesn't want to be pinged when you Reply to their messages! "
                "https://tenor.com/view/discord-gif-9382638485807678151", mention_author=True)
            # one auto silent bad bot, so people maxxing negative credit don't spam this
            await updateCredit(msg.author.id, credit=-15, treason=1)

    # De-obscure links first, forgoing all other features (we want to discourage hiding links)
    linkSearch = re.search(r"\[(.+)\]\((\w+://)((?:[a-z0-9-]+\.)+\w+)(.*)\)", contentL)
    if linkSearch and linkSearch.group(3) != "discord.com":
        await msg.reply(
            "Please don't mask links with markdown in this server, thanks! That link leads to "
            f"{linkSearch.group(3)}{linkSearch.group(4)}", mention_author=True)
        # one auto silent bad bot, so people maxxing negative credit don't spam this
        await updateCredit(msg.author.id, credit=-15, treason=1)

    # Check good/bad/"any" bots - can have any leading/trailing words as long as 2nd one is "bot"
    # Or can start with "good"/"bad"/"medium" and have precisely one trailing word
    elif (len(splitL) > 1 and splitL[1] == "bot") or (
            len(splitL) == 2 and splitL[0] in ["good", "bad", "medium"]):
        # Let sentiment analysis decide whether vote is positive or negative
        sentiment = sentimentAnalysis(msg.content)
        # good/bad bots are worth +/-3 credit base, so add 5x mult before channel mult
        # so manual votes are worth 15 credit before channel mult. This still seems kinda devalued
        # compared to Sbeve votes, but there's no vote limit now so maybe it balances out
        credit = convertSentiment(sentiment) * 5 * creditMult
        target = await findVoteTarget(msg)
        if target is not None:
            name = getAuthorMember(target).nick
            await updateCredit(target.author.id, credit=credit, name=name)
            #TODO if quietRole[msg.guild] in target.author.roles:
            #     await target.channel.send(
            #         f"Thank you for voting on {name}! "
            #         f"They now have {userData[target.author.id]["credit"]} social credit.")
            # else:
            #     await target.channel.send(
            #         f"Thank you for voting on {target.author.mention}! "
            #         f"They now have {userData[target.author.id]["credit"]} social credit.")

    # Scan for short message triggers, if present handle those and return without further processing
    # Also results in messages needing to be a minimum length of 25 chars for sentiment analysis.
    # Exclude messages of 0 length since Discord considers some system/thread messages as 0-length
    elif 1 <= len(contentL) <= 25:
        strippedContent = contentL.strip("^?!")
        if strippedContent in config["wordlistTime"]:
            if now.strftime("%a") == "Wed":
                await msg.channel.send("It is Wednesday, my dudes")
            else:
                await msg.channel.send(
                    f"It is currently {now.strftime("%H:%M %Z")}!")

        # If stripped msg is in the what wordlist or is empty, trigger bot's "what" response
        # Exclude messages with embeds/attachments/references as user might be reacting to those
        elif len(msg.attachments) == 0 and len(msg.embeds) == 0 and msg.reference is None and (
                strippedContent in config["wordlistWhat"] or not strippedContent):
            try:
                # Convert recent channel history into list of messages
                history = [oldMsg async for oldMsg in msg.channel.history(limit=2)]
                # Strip existing bolding from the target msg, so we can make the response be bolded
                target = history[-1].content.replace("**","").upper() # all caps letsgo
                await msg.channel.send(f"**{target}**")
            except Exception as e:
                await msg.channel.send(
                    "**I TRIED TO RETRIEVE THE PREVIOUS MESSAGE TO CAPITALIZE IT BUT I BROKE INSTEAD**")

        # If message contains the word "1984", censor it with minimal delay
        elif "1984" in splitL:
            await msg.delete(delay=8)

    # No special handling, just do sentiment analysis and modify user's social credit
    elif len(contentL) > 25:
        # Invert if discussing treasonous things (saying bad things about bad things is now good)
        # Use fuzzy matching on treason wordlist, apparently it only takes 4ms for >2000-char msg
        # to be checked against a list of 10 treasonous words on gamer rig
        fuzzlist = [
            fuzz.ratio(word, msgWord) > 80
            for msgWord in splitL for word in config["wordlistTreason"]]
        treasonCount = fuzzlist.count(True)
        if treasonCount > 0:
            creditMult += treasonCount//4 # if they're really going ham on the matches, increase mult
            creditMult *= -1

        # Get base message sentiment and credit value, and modify user's credit
        sentiment = sentimentAnalysis(msg.content)
        credit = convertSentiment(sentiment) * creditMult
        await updateCredit(
            msg.author.id, credit=credit, treason=min(1, treasonCount),
            date=now.strftime("%Y-%m-%d"))

        # roughly 10% chance to save current data to disk, in case of power failure, etc.
        # yes i know it's not actually a 10% chance due to message IDs being snowflakes but shhh
        if msg.id % 10 == 0:
            await writeCreditToDisk()


def sentimentAnalysis(content: str) -> float:
    """Analyzes sentiment of {content}, which is a float between 0 (negative) and 1 (positive).
    The ^vibe command outputs this value, and convertSentiment (into credit) uses this as input."""
    tokenizedInput = numpy.array([tokenizer.encode(content)], dtype=numpy.int32)
    tokenizedInput = keras.preprocessing.sequence.pad_sequences(
        tokenizedInput, value=0, padding="post", maxlen=73)

    tfInterp.set_tensor(tfInput[0]["index"], tokenizedInput)
    tfInterp.invoke()
    tfSent = tfInterp.get_tensor(tfOutput[0]["index"])
    # h5Sent = h5Model(tokenizedInput, training=False).numpy()

    return tfSent


def convertSentiment(sentiment: float) -> int:
    """{sentiment} is a value from 0 to 1, where 0 indicates strongly negative sentiment and 1
    indicates strongly positive. This function converts sentiment into a social credit value,
    where 0.32 to 0.68 inclusive is neutral and worth 0 credit. Social credit is granted over 0.68,
    and subtracted below 0.32, with greater credit assigned as sentiment approaches 0 or 1."""
    for i in range(1,6):
        # Increased credit on log2 scale. Basically, assign -(6-x) credit (where 1 <= x <= 5) if the
        # difference between {sentiment}*100 and 0 is less than 2^x but greater/equal to 2^(x-1)
        if sentiment*100 < 2**i:
            return i-6
        # Likewise, but assign 6-x credit, and check the difference between {sentiment}*100 and 100
        # If sentiment is ever somehow less than 0 or greater than 1, credit maxes at -5/+5
        elif sentiment*100 > 100 - 2**i:
            return 6-i
    return 0


async def findVoteTarget(msg: discord.Message) -> discord.Message | None: #TODO testing
    """Given {msg}, find the target message that {msg} is voting on. If {msg} is a reply, then
    this is very easy, but limit voting in this way to messages sent within the past day, and only
    once, to avoid infinite vote chains on the same message or on all prior messages sent by a user.

    If {msg} is not a reply, look through channel history for the first message that isn't a bot
    response to a vote, nor is another user voting the same message."""
    # Implement vote-replying, and subsequent chained votes will continue to vote on the same target
    if msg.reference is not None:
        trgtMsg = await msg.channel.fetch_message(msg.reference.message_id)
        if trgtMsg.created_at.day >= msg.created_at.day - 1 and not any(
                reaction.me for reaction in trgtMsg.reactions):
            # Reply-voting bypasses voting-on-self guards, so check it here
            if trgtMsg.author.id == msg.author.id:
                #TODO await msg.reply(
                    # "You can only vote on messages sent by other users!", mention_author=False)
                return None
            else:
                # add a reaction to target to track reply-vote usage, then return it
                await trgtMsg.add_reaction("\N{eye}")
                return trgtMsg
        else:
            #TODO await msg.reply(
            #     "You can only vote via reply on messages sent since 12 AM yesterday, and messages "
            #     "can only get reply-voted once (follow-up votes will continue that vote chain).",
            #     mention_author=False)
            return None

    else:
        first = True
        derefMsg = None
        async for trgtMsg in msg.channel.history(limit=32):
            # Skip the user's own vote that triggered this method
            if first:
                first = False
                continue

            # If current trgtMsg is a reply, keep track of the message it's replying to
            elif not trgtMsg.author.bot and trgtMsg.reference is not None:
                derefMsg = await msg.channel.fetch_message(trgtMsg.reference.message_id)

            # Check if this trgtMsg is a vote or the bot's response to a vote
            contentL = trgtMsg.content.lower()
            splitL = contentL.split()
            if (len(splitL) > 1 and splitL[1] == "bot") or (
                    len(splitL) == 2 and splitL[0] in ["good", "bad", "medium"]) or (
                    trgtMsg.author.bot and
                    re.search(r"^(?:thank )?you (?:for|can) (?:voting|only) (?:on|vote) ", contentL)):

                # If the voter is trgtMsg's author, voter already voted on the message
                if trgtMsg.author.id == msg.author.id:
                    #TODO await msg.reply("You can only vote on a message once!", mention_author=False)
                    return None

                # If trgtMsg is a vote or response and derefMsg exists, the target is derefMsg:
                # this block enters the first time a vote is cast that's replying to derefMsg (since
                # derefMsg can't be set by a bot, a reply-vote will always enter this block first).
                # If trgtMsg is NOT a vote or response, the parent block doesn't enter, so we can
                # still vote on messages that happen to be replies (and not their reference message)
                elif derefMsg is not None:
                    # Follow-up votes on a reply-vote bypass voting-on-self guards, so check it here
                    if derefMsg.author.id == msg.author.id:
                        #TODO await msg.reply(
                            # "You can only vote on messages sent by other users!",
                            # mention_author=False)
                        return None
                    else:
                        return derefMsg

                continue

            # Skip the author's own messages, so they can talk and then vote on the prior message
            elif trgtMsg.author.id == msg.author.id:
                continue

            else:
                return trgtMsg

    return None # Only get here if the history limit is reached, or multiple users are reply-voting


def getAuthorMember(msg: discord.Message) -> discord.Member:
    """Given a message, return the discord.Member object representing the message's author.
    Needed because in private channels, msg.author returns a User instead of a Member,
    but other parts of the code need the additional fields that Member contains."""
    return msg.guild.get_member(msg.author.id)


async def updateCredit(
    userID: int, credit: int=0, treason: int=0, name: str=None, date: str=None) -> None:
    """Update user social credit in async-safe way. {credit} should be a positive or negative
    integer - the amount by which to modify the user's social credit. Likewise for {treason}.
    {name} and {date} are purely informational fields, which get populated over time, to help
    users read the stats file and make sense of it at a glance."""
    userID = str(userID) # Convert to string, because the keys in the stats file are all strings
    async with dataLock:
        global userData
        if userID not in userData.keys():
            userData[userID] = {"credit": 0, "treason": 0}
        userData[userID]["credit"] += credit
        userData[userID]["treason"] += treason
        if name is not None:
            userData[userID]["name"] = name
        if date is not None:
            userData[userID]["lastUpdate"] = date


async def readCreditFromDisk() -> None:
    """Retrieve all user social credit values from on-disk data file. Should only be needed on
    bot startup."""
    filePath = Path(f"{sys.path[0]}/data/user-stats.json")
    try:
        async with dataLock:
            with open(filePath) as file:
                global userData
                userData = json.load(file)

    except FileNotFoundError:
        filePath.touch()

    except json.decoder.JSONDecodeError:
        journal.write(
            "Couldn't load user stats from file, this is expected if the file is empty.",
            priority=journal.Priority.WARNING)
        with open(filePath) as file:
            if len(file.read()) > 20: # there's at least one user entry
                sys.exit(
                    "User-stats is non-empty, but couldn't load, so data must be corrupted.\n"
                    "PLEASE ATTEMPT TO RECOVER THE DATA.")


async def writeCreditToDisk() -> None:
    """Write current social credit values for all users to the on-disk data file. Since almost every
    message will cause a social credit update, this method shouldn't be called by every on_message.
    Scheduling is cool, but it's more fun to give on_message a low chance to call this method."""
    filePath = Path(f"{sys.path[0]}/data/user-stats.json")
    try:
        async with dataLock:
            # 5% chance to take a backup first
            if random.random() > 0.95:
                backupPath = Path(f"{sys.path[0]}/data/user-stats.json.bak")
                with open(backupPath) as backup:
                    if len(backup.read()) > 20: # there's at least one user entry
                        # move cursor back to start of file, and load the backup(s) from disk
                        backup.seek(0)
                        prevBackup = json.load(backup)
                    else:
                        prevBackup = {}

                # Create (then rotate through) 3 timestamped backups
                num = len(prevBackup.keys())
                now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                with open(backupPath, "w") as backup:
                    if num < 3:
                        prevBackup[f"{num + 1}"] = {}
                        prevBackup[f"{num + 1}"]["timestamp"] = now
                        prevBackup[f"{num + 1}"]["userdata"] = userData
                    else:
                        # Look for signs of abuse - ridiculous credit gain/loss since last backup
                        # Invert user's credit if >2000 difference and give 10 treason
                        for user in prevBackup["3"]["userdata"]:
                            if (
                                    userData[user]["credit"] > prevBackup["3"]["userdata"][user]["credit"] + 2000 or
                                    userData[user]["credit"] < prevBackup["3"]["userdata"][user]["credit"] - 2000):
                                userData[user]["credit"] *= -1
                                userData[user]["treason"] += 10

                        # Take the backup - first two operations just reassign pointers,
                        # so need to initialize ["3"] to a new dict to not update ["2"]'s values
                        prevBackup["1"] = prevBackup["2"]
                        prevBackup["2"] = prevBackup["3"]
                        prevBackup["3"] = {}
                        prevBackup["3"]["timestamp"] = now
                        prevBackup["3"]["userdata"] = userData

                    json.dump(prevBackup, backup, indent=4)

            # Write user data to disk
            with open(filePath, "w") as file:
                json.dump(userData, file, indent=4)

    except Exception as e:
        journal.write(
            f"Either couldn't take backup, or couldn't write user credit to disk, due to: {e}\n\n"
            "If the backup failed, the main file also wasn't written, to preserve current contents."
            f"\n\nDumping current user data:{userData}", priority=journal.Priority.ERROR)


if __name__ == "__main__":
    # loop = asyncio.new_event_loop()
    # loop.run_forever(client.run(userInfo["authInfo"]["discordToken"]))
    client.run(userInfo["authInfo"]["discordToken"])
