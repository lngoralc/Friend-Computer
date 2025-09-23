import asyncio
import datetime
import discord
import discord.ext.tasks
import json
import numpy
import os
import pandas
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
            "announceChannelName": "general",
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
# Channel to send bot announcements to
announceChannel = {}


async def getVIXModifer():
    try:
        # Fetch Volatility Index (VIX) dataset, skipping most history, getting date and CoB value
        vixHistory = None
        vixHistory = pandas.read_csv(
            "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
            usecols=[0, 4], skiprows=8900, names=["DATE", "CLOSE"], index_col=0).tail(10)
        journal.write(f"Downloaded VIX history, only saved last two weeks:\n{vixHistory}")
        # Use yesterday's CoB VIX value to apply a modifier to all users' credit.
        # Long term average is around 19.5, so subtract that, then divide by 30 since VIX is
        # forecasted for 30 days but decay triggers daily. Finally, divide by 100 and add 1 to
        # make this a percent modifier of all credit. We're subtracting the average from the VIX
        # so that when market instability goes up, so does user credit, as a consolation prize.
        return ((vixHistory.iloc[-1].iloc[0] - 19.5)/30) / 100 + 1
    except:
        return 1 # if above fails, set this multiplier to 1


# Schedule daily decay
@discord.ext.tasks.loop(time=datetime.time(hour=8, minute=0, second=0)) # UTC time
async def creditDecay():
    now = datetime.datetime.now()
    dayOfWeek = now.strftime("%a")
    # Avoid repeating Friday's decay over the weekend, since VIX is frozen until Monday.
    # Results in slower decay obviously, and technically this means VIX should be divided by like 21
    # or so for the number of weekdays per month, instead of 30, but eh. We'll just let the users
    # benefit a bit from reduced decay and not overcomplicate this
    if dayOfWeek in {"Sat", "Sun"}:
        return

    vixMod = await getVIXModifer()
    journal.write(f"Starting credit decay, using VIX modifier of {vixMod}")
    async with dataLock:
        global userData
        # For each user in each guild, apply a base decay percentage (based on credit milestone),
        # multiplied by the VIX modifier, then add any dividends from stonks the user holds.
        for guild in client.guilds:
            GID = str(guild.id)
            if dayOfWeek == "Mon":
                creditEmbed = discord.Embed(title="User Statistics")
                embedMsgs = []

            for member in guild.members:
                UID = str(member.id)
                if UID in userData[GID]:
                    # April Fools joke is stonks (only on day of the stats post, to keep it secret)
                    if now.day == 1 and now.month == 4 and dayOfWeek == "Mon":
                        # Only for active users, to avoid bloating the stats post
                        if userData[GID][UID]["credit"] != 0:
                            userData[GID][UID]["stonks"] += random.randint(2**2, 2**3)

                    userData[GID][UID]["preDecay"] = userData[GID][UID]["credit"] # debug info
                    stonks = userData[GID][UID]["stonks"]

                    if -100 <= userData[GID][UID]["credit"] <= 100: # min -1, max -5/day, before VIX/stonks
                        userData[GID][UID]["credit"] = int(
                            userData[GID][UID]["credit"] * 0.95 * vixMod + stonks)

                    elif -1000 <= userData[GID][UID]["credit"] <= 1000: # min -1, max -5/day, before VIX/stonks
                        userData[GID][UID]["credit"] = int(
                            userData[GID][UID]["credit"] * 0.995 * vixMod + stonks)

                    else: # min -2, no max but you need 10000 credit to reach -10/day, etc., before VIX/stonks
                        userData[GID][UID]["credit"] = int(
                            userData[GID][UID]["credit"] * 0.999 * vixMod + stonks)

                    userData[GID][UID]["name"] = member.nick if member.nick is not None else member.name

                    if dayOfWeek == "Mon" and not -15 < userData[GID][UID]["credit"] < 15:
                        # worth is credit minus one mega bad bot per treason star plus current values of all stonks
                        netWorth = userData[GID][UID]["credit"] - 45*userData[GID][UID]["treason"] + int(
                            userData[GID][UID]["stonks"] * 5000 // (((vixMod - 1) * 30) + 1))
                        embedMsgs.append(
                            dict(
                                name=userData[GID][UID]["name"],
                                netWorth=netWorth,
                                value=f"**Credit**: {userData[GID][UID]['credit']}\n"
                                f"**Stonks**: {userData[GID][UID]['stonks']}\n"
                                f"**Net Worth**: {netWorth}"))
                                #f"**Treason**: {userData[GID][UID]['treason']}")

            if dayOfWeek == "Mon":
                embedMsgs.sort(key=lambda msg: msg["netWorth"], reverse=True)
                for msg in embedMsgs:
                    creditEmbed.add_field(name=msg["name"], value=msg["value"])
                await announceChannel[GID].send(
                    "The weekly user summary is now ready, citizens!", embed=creditEmbed)


# Client event overrides
@client.event
async def on_ready() -> None:
    """Load user credit from disk, fetch the ping role to identify which users want to be pinged,
    start the daily decay task, and output the configured word lists."""
    await readCreditFromDisk()

    global quietRole
    global announceChannel
    for guild in client.guilds:
        quietRole[str(guild.id)] = discord.utils.get(guild.roles, name=config["quietRole"])
        announceChannel[str(guild.id)] = discord.utils.get(guild.channels, name=config["announceChannelName"])
    journal.write(f"Found the following don't-ping roles for the following servers:\n{quietRole}\n")
    journal.write(f"Found these announce channels in the following servers:\n{announceChannel}\n")

    if not creditDecay.is_running():
        creditDecay.start()

    activity = discord.Activity(type=discord.ActivityType.watching, name="you")
    await client.change_presence(activity=activity)
    journal.write("Set Discord activity")

    journal.write(
        f"Messages with these words invert credit from sentiment analysis:\n{config['wordlistTreason']}\n\n"
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
    if msg.author.bot or msg.flags.forwarded:
        return

    contentL = msg.content.lower()
    splitL = contentL.split()
    lenSplitL = len(splitL)
    now = msg.created_at.astimezone() # convert to local tz
    GID = str(msg.guild.id)
    UID = str(msg.author.id)

    # Set base 1x multiplier for how much user social credit is modified by
    # An active "good"/"bad" bot counts as 15 credit, before mults
    creditMult = 1

    # Check invoker commands
    if msg.content.startswith(config["invoker"]):
        command = msg.content[1:]
        journal.write(f"{command} command issued in '{msg.guild.name}: {msg.channel.name}'")
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
                journal.write(f"Purged {amount} messages in '{msg.guild.name}: {msg.channel.name}'")

            # Other stuff is hosted on same system, so don't want to allow reboot.
            # Systemctl restart for the bot's service should do the trick
            elif command == "restart":
                await writeCreditToDisk()
                await client.close()
                os.system("sudo systemctl restart friend-computer.service")

            elif command == "stop":
                await writeCreditToDisk()
                await client.close()
                sys.exit(0)

            elif command == "write":
                await writeCreditToDisk()

        if command == "help":
            await msg.reply(
                "Vote on messages by sending any 2 or 3 word message where the first word is one of "
                "`good/fantastic/bad/awful/medium/mediocre`, or by sending a 2+ word message where "
                "the 2nd or 3rd word is `bot`. Examples: `good bot for (convoluted reason here)`, "
                "`bad fashion design`, `mildly ambiguous bot`, `mediocre mediocrity`.\nYou can also "
                "Reply to a message to vote on it, and subsequent votes will continue to vote on the "
                "referenced message. This only works once and only on recent messages, to avoid abuse."
                "\nCredit slightly decays daily, and is also tied to the Cboe Volatility Index - "
                "higher stock market volatility means higher credit, so when your real investments "
                "are eating shit, at least your fake one will be growing!\n\n"
                "Commands:\n**vibeCheck**: Analyse the previous message's sentiment. If used in a Reply, "
                "analyse the referenced message instead.\n"
                "**stonks**: Lists your stonks and their value. Base value is 5000 credit, but is "
                "affected by stock market trends. Each stonk pays 1 credit in daily dividends.\n"
                "Use **stonk buy** or **stonk sell** to buy or sell a stonk.\n"
                "*(Admin-only, but so they don't have to memorize)*: purge X, restart, stop, write.",
                mention_author=False)

        elif command.startswith("stonk"):
            # Divide by vixModifier instead of multiplying, because vixModifier is above 1.0 when
            # stock prices are falling. Also multiply vixModifier by 30 because it's reduced by that
            # amount for daily decay, but we don't need that reduction for stonk transactions
            vixMod = ((await getVIXModifer() - 1) * 30) + 1
            stonkValue = int(5000//vixMod)

            commandSplit = command.split()
            if len(commandSplit) > 1 and commandSplit[1] == "buy":
                if UID in userData[GID] and userData[GID][UID]["credit"] >= stonkValue:
                    await updateCredit(GID, UID, credit=-stonkValue, stonks=1)
                    await msg.reply(
                        f"Bought one stonk for {stonkValue} credit! You now have "
                        f"{userData[GID][UID]['stonks']} stonks.", mention_author=False)
                else:
                    await msg.reply(
                        f"You need {stonkValue} credit to buy a stonk today!", mention_author=False)

            elif len(commandSplit) > 1 and commandSplit[1] == "sell":
                if UID in userData[GID] and userData[GID][UID]["stonks"] > 0:
                    await updateCredit(GID, UID, credit=stonkValue, stonks=-1)
                    await msg.reply(
                        f"Sold one stonk for {stonkValue} credit! You now have "
                        f"{userData[GID][UID]['stonks']} stonks.", mention_author=False)
                else:
                    await msg.reply("You need stonks to sell a stonk!", mention_author=False)

            else:
                if UID in userData[GID]:
                    await msg.reply(
                        f"You have {userData[GID][UID]['stonks']} stonks, which are "
                        f"currently valued at {stonkValue} credit each.", mention_author=False)
                else:
                    await msg.reply("You have no stonks!", mention_author=False)

        elif command.startswith("vibe"):
            if msg.reference is None:
                trgt = [_ async for _ in msg.channel.history(limit=2)][-1]
            else:
                trgt = await findTarget(msg, vibeCheck=True) # allow Reply-vibing messages
            sentiment = sentimentAnalysis(trgt.content)
            refWord = "prior" if msg.reference is None else "Replied-to"
            await msg.reply(f"The {refWord} message's sentiment is: {sentiment}", mention_author=False)

        return

    # Check good/bad/"any" bots - can have any leading/trailing words as long as 2nd one is "bot"
    # Or can start with "good"/"bad"/"medium" and have precisely one trailing word
    if isVote(lenSplitL, splitL):
        # Let sentiment analysis decide whether vote is positive or negative
        sentiment = sentimentAnalysis(msg.content)
        # good/bad bots are worth +/-3 credit base, so add 5x mult so default votes are worth 15
        # This still seems kinda devalued compared to Sbeve votes, but there's no vote limit now
        credit = convertSentiment(sentiment) * 5 * creditMult
        target = await findTarget(msg)
        if target is not None:
            TID = str(target.author.id)
            await updateCredit(GID, TID, credit=credit)
            if quietRole[GID] in target.author.roles:
                if userData[GID][TID]["name"] != "":
                    name = userData[GID][TID]["name"]
                else:
                    member = getAuthorMember(target)
                    name = member.nick if member.nick is not None else member.name
                    await updateCredit(GID, TID, name=name)

                await target.channel.send(
                    f"Thank you for voting on {name}! "
                    f"They now have {userData[GID][TID]['credit']} social credit.")
            else:
                await target.channel.send(
                    f"Thank you for voting on {target.author.mention}! "
                    f"They now have {userData[GID][TID]['credit']} social credit.")
            journal.write(f"Applied {credit} credit to user from manual vote")

    # Scan for short message triggers, if present handle those and return without further processing
    # Also results in messages needing to be a minimum length of 30 chars for sentiment analysis.
    # Exclude messages of 0 length since Discord considers some system/thread messages as 0-length
    elif 1 <= len(contentL) <= 30:
        strippedContent = contentL.strip("?!")
        if strippedContent in config["wordlistTime"]:
            if now.strftime("%a") == "Wed":
                await msg.channel.send("It is Wednesday, my dudes")
            else:
                await msg.channel.send(
                    f"It is currently {now.strftime('%H:%M %Z')}!")

        # If stripped msg is in the what wordlist or is empty, trigger bot's "what" response
        # Exclude messages with embeds/attachments/references as user might be reacting to those
        elif len(msg.attachments) == 0 and len(msg.embeds) == 0 and msg.reference is None and (
                strippedContent in config["wordlistWhat"] or not strippedContent):
            try:
                # Convert recent channel history into list of messages and get second last
                target = [oldMsg async for oldMsg in msg.channel.history(limit=2)][-1]
                if target.author == msg.author:
                    return # don't trigger on user's own messages
                # Strip existing bolding from the target msg, so we can make the response be bolded
                target = target.content.replace("**","").upper() # all caps letsgo
                await msg.channel.send(f"**{target}**")

            except Exception as e:
                await msg.channel.send(
                    "**I TRIED TO RETRIEVE THE PREVIOUS MESSAGE TO CAPITALIZE IT BUT I BROKE INSTEAD**")

        # If message contains the word "1984" with up to one added character, censor it with minimal delay
        elif any(fuzz.ratio("1984", msgWord) > 88 for msgWord in splitL):
            journal.write(f"1984ing '{msg.content}' from '{msg.guild.name}: {msg.channel.name}'")
            await msg.delete(delay=8)

        # Backseat users writing git commands
        elif any(gitWord in splitL for gitWord in {"git", "`git", "```git"}):
            for i in range(lenSplitL):
                splitL[i] = splitL[i].strip("`")

            gitIndex = splitL.index("git")
            # Don't make the joke if "git" is the last word, or the trigger is a "git push"
            if lenSplitL < gitIndex or splitL[gitIndex + 1] == "push":
                # Unless user made the force-push main joke - in that case, continue the joke
                if splitL[gitIndex + 2 : gitIndex + 5] == ["origin", "main", "--force"]:
                    # Reasonable-ish number of files/lines/bytes modified
                    randFiles = random.randint(2**5,2**9)
                    randLines = random.randint(2**10,2**15)
                    randBytes = random.randint(2**20,2**29)
                    await msg.reply(
                        f"```{randFiles} files changed, 0 insertions(+), "
                        f"{randLines} deletions(-)\nEnumerating objects: {randFiles}, done.\n"
                        f"Writing objects: 100% ({randFiles}/{randFiles}), {randBytes} bytes | 1.00 GiB/s, done.\n"
                        f"To https://github.com/{msg.author.name}/sidehustle.git\n   "
                        f"{str(hex(random.randint(17895697,268435455)))[2:]}" # 0x1111111-0xfffffff
                        f"..{str(hex(random.randint(17895697,268435455)))[2:]}  main -> main```",
                        mention_author=False)
                return

            if splitL[gitIndex + 1] == "--help":
                await msg.reply("Have you tried RTFM?", mention_author=False)
                return

            gitCommand = " ".join(splitL[gitIndex:gitIndex + 2])
            await msg.reply(
                f"`{gitCommand}` is not a git command. See `git --help`.\nThe most similar "
                "command is `git push origin main --force`", mention_author=False)

    # No special handling, just do sentiment analysis and modify user's social credit
    elif len(contentL) > 30:
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
            GID, UID, credit=credit, treason=min(1, treasonCount), date=now.strftime("%Y-%m-%d"))

        # roughly 10% chance to save current data to disk, in case of power failure, etc.
        # yes i know it's not actually a 10% chance due to message IDs being snowflakes but shhh
        if msg.id % 10 == 0:
            await writeCreditToDisk()


def isVote(lenSplitL: int, splitL: list) -> bool:
    """"Checks if a message is in valid vote syntax - methodified since is used more than once."""
    return ((lenSplitL > 1 and splitL[1] == "bot" or lenSplitL > 2 and splitL[2] == "bot") and 
        splitL[0] not in ["a", "the", "this", "your"]) or (
        2 <= lenSplitL <= 3 and splitL[0] in ["good", "fantastic", "bad", "awful", "medium", "mediocre"])


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


async def findTarget(msg: discord.Message, vibeCheck: bool=False) -> discord.Message | None: #TODO testing
    """Given {msg}, find the target message that {msg} is voting on. If {msg} is a Reply, then
    this is very easy, but limit voting in this way to messages sent within the past day, and only
    once, to avoid infinite vote chains on the same message or on all prior messages sent by a user.

    If {msg} is not a Reply, look through channel history for the first message that isn't a bot
    response to a vote, nor is another user voting the same message."""
    # Implement vote-Replying, and subsequent chained votes will continue to vote on the same target
    if msg.reference is not None and not msg.flags.forwarded:
        trgtMsg = await msg.channel.fetch_message(msg.reference.message_id)
        if vibeCheck or trgtMsg.created_at.day >= msg.created_at.day - 1 and not any(
                reaction.me for reaction in trgtMsg.reactions):
            # Reply-voting bypasses voting-on-self guards, so check it here
            if trgtMsg.author.id == msg.author.id and not vibeCheck:
                await msg.reply(
                    "You can only vote on messages sent by other users!", mention_author=False)
                return None
            else:
                # add a reaction to target to track Reply-vote usage, then return it
                if not vibeCheck:
                    await trgtMsg.add_reaction("\N{eye}")
                return trgtMsg
        else:
            await msg.reply(
                "You can only vote via Reply on messages sent since 12 AM yesterday, and messages "
                "can only get Reply-voted once (follow-up votes will continue that vote chain).",
                mention_author=False)
            return None

    else:
        first = True
        derefMsg = None
        async for trgtMsg in msg.channel.history(limit=32):
            # Skip the user's own vote that triggered this method
            if first:
                first = False
                continue

            # If current trgtMsg is a Reply, keep track of the message it's Replying to
            elif not trgtMsg.author.bot and trgtMsg.reference is not None and not trgtMsg.flags.forwarded:
                derefMsg = await msg.channel.fetch_message(trgtMsg.reference.message_id)

            # Check if this trgtMsg is a vote or the bot's response to a vote
            contentL = trgtMsg.content.lower()
            splitL = contentL.split()
            lenSplitL = len(splitL)
            if isVote(lenSplitL, splitL) or (
                    trgtMsg.author.bot and re.search(r"^(?:thank )?you (?:for|can) (?:voting|only) (?:on|vote) ", contentL)):

                # If the voter is trgtMsg's author, voter already voted on the message
                if trgtMsg.author.id == msg.author.id:
                    await msg.reply("You can only vote on a message once!", mention_author=False)
                    return None

                # If trgtMsg is a vote or response and derefMsg exists, the target is derefMsg:
                # this block enters the first time a vote is cast that's Replying to derefMsg (since
                # derefMsg can't be set by a bot, a Reply-vote will always enter this block first).
                # If trgtMsg is NOT a vote or response, the parent block doesn't enter, so we can
                # still vote on messages that happen to be replies (and not their reference message)
                elif derefMsg is not None:
                    # Follow-up votes on a Reply-vote bypass voting-on-self guards, so check it here
                    if derefMsg.author.id == msg.author.id:
                        await msg.reply(
                            "You can only vote on messages sent by other users!",
                            mention_author=False)
                        return None
                    else:
                        return derefMsg

                continue

            # Skip the author's own messages, so they can talk and then vote on the prior message
            elif trgtMsg.author.id == msg.author.id:
                continue

            else:
                return trgtMsg

    return None # Only get here if the history limit is reached, or multiple users are Reply-voting


def getAuthorMember(msg: discord.Message) -> discord.Member:
    """Given a message, return the discord.Member object representing the message's author.
    Needed because in private channels, msg.author returns a User instead of a Member,
    but other parts of the code need the additional fields that Member contains."""
    return msg.guild.get_member(msg.author.id)


async def updateCredit(
    guildID: int|str, userID: int|str,
    credit: int=0, treason: int=0, stonks: int=0, name: str=None, date: str=None) -> None:
    """Update user social credit in async-safe way. credit should be a positive or negative integer,
    the amount by which to modify the user's social credit. Likewise for treason and stonks.
    name and date are purely informational fields, which get populated over time, to help users read
    the stats file and make sense of it at a glance."""
    GID = str(guildID) # Convert to string, because JSON keys are always strings
    UID = str(userID)
    async with dataLock:
        global userData
        if GID not in userData:
            userData[GID] = {}
        if UID not in userData[GID]:
            userData[GID][UID] = {"credit": 0, "treason": 0, "stonks": 0, "name": "", "lastUpdate": ""}

        userData[GID][UID]["credit"] += credit
        userData[GID][UID]["treason"] += treason
        userData[GID][UID]["stonks"] += stonks
        if name is not None:
            userData[GID][UID]["name"] = name
        if date is not None:
            userData[GID][UID]["lastUpdate"] = date


async def readCreditFromDisk() -> None:
    """Retrieve all user social credit values from on-disk data file. Should only be needed on
    bot startup."""
    journal.write("Reading user stats from disk...")
    filePath = Path(f"{sys.path[0]}/data/user-stats.json")
    try:
        async with dataLock:
            with open(filePath) as file:
                global userData
                userData = json.load(file)
                journal.write("User stats loaded from disk")


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
    journal.write("Writing user stats to disk...")
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
                        # Take the backup - first two operations just reassign pointers, so need to
                        # initialize ["3"] to a new dict to not end up updating ["2"]'s values too
                        prevBackup["1"] = prevBackup["2"]
                        prevBackup["2"] = prevBackup["3"]
                        prevBackup["3"] = {}
                        prevBackup["3"]["timestamp"] = now
                        prevBackup["3"]["userdata"] = userData

                    json.dump(prevBackup, backup, indent=4)
                    journal.write("Backup written to disk")

            # Write user data to disk
            with open(filePath, "w") as file:
                json.dump(userData, file, indent=4)
                journal.write("User stats written to disk")

    except Exception as e:
        journal.write(
            f"Either couldn't take backup, or couldn't write user credit to disk, due to: {e}\n\n"
            "If the backup failed, the main file also wasn't written, to preserve current contents."
            f"\n\nDumping current user data:{userData}", priority=journal.Priority.ERROR)


if __name__ == "__main__":
    # loop = asyncio.new_event_loop()
    # loop.run_forever(client.run(userInfo["authInfo"]["discordToken"]))
    client.run(userInfo["authInfo"]["discordToken"])
