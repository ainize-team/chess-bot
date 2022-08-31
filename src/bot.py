import io
import os
from collections import defaultdict

import chess
import discord
import torch
from discord import app_commands
from discord.ui import Button, View
from PIL import Image
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from generator import Generator


# TO-DO: 체스를 잘 둘 수 있는 모델 학습 및 구현(RNN 등을 이용)
# Fine-tunning 모델 로드
torch.manual_seed(42)
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>"
)
configuration = GPT2Config.from_pretrained("gpt2.pth", output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2.pth", config=configuration)
model.resize_token_embeddings(len(tokenizer))


# 디스코드 봇 로그인
class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.synced = False

    async def on_ready(self):
        await self.wait_until_ready()
        if not self.synced:
            await tree.sync(guild=discord.Object(id=int(os.environ.get("GUILD_ID"))))
            self.synced = True
        print(f"We have logged in as {self.user}.")


client = aclient()
tree = app_commands.CommandTree(client)


# 모델을 이용해 prompt의 다음 토큰 생성
def get_next_move(prompt):
    # prompt 이후 5번 Text Generation
    # e.g) 1. e4 -> 1. e4 e5 ...
    generated = tokenizer("<|startoftext|>" + " ".join(prompt), return_tensors="pt").input_ids
    sample_outputs = model.generate(
        generated,
        do_sample=True,
        top_k=50,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        max_length=300,
        top_p=0.95,
        temperature=1.9,
        num_return_sequences=5,
    )

    # 많이 나온 순으로 토큰 정렬
    # e.g) {'Nh3': 3, 'e4': 1, 'd4': 1}
    legal_move_count = defaultdict(int)
    for sample_output in sample_outputs:
        move = tokenizer.decode(sample_output, skip_special_tokens=True)[len(" ".join(prompt)) + 1 :]
        legal_move_count[move[: move.index(" ")]] += 1

    return sorted(legal_move_count.items(), key=lambda x: -x[1])


# 체스판을 시각화
def display_board(USER_CHESS):
    with io.BytesIO() as binary:
        board = Generator.generate(USER_CHESS).resize((500, 500), Image.ANTIALIAS)
        board.save(binary, "PNG")
        binary.seek(0)
        file = discord.File(fp=binary, filename="board.png")

    return file


async def human_move(interaction, move):
    # 사용자가 /move로 입력한 move를 추가
    user_chess[interaction.user.id].push_san(move)
    user_prompt[interaction.user.id].append(move)

    embed = discord.Embed(
        title="Your game with Chess Bot",
        description="`" + str(interaction.user.name) + "` have moved: **" + move + "**",
        color=discord.Color.green(),
    )
    return embed, display_board(user_chess[interaction.user.id])


async def bot_move(interaction, embed, file):
    await interaction.response.defer(ephemeral=True)  # response 이전까지 chess-bot is thinking... 문구 표시 (Discord)
    moves = get_next_move(user_prompt[interaction.user.id])
    await interaction.followup.send(embed=embed, file=file, ephemeral=True)  # 사용자의 move 표시 (Discord)

    legal_move = str(user_chess[interaction.user.id].legal_moves)
    legal_move = legal_move[legal_move.index("(") + 1 : legal_move.index(")")].split(", ")

    # 생성한 chess-bot의 move를 추가
    move_success = 0
    for move in moves:
        if move[0] in legal_move:
            user_chess[interaction.user.id].push_san(move[0])
            user_prompt[interaction.user.id].append(move[0])
            move_success = 1
            break

    # chess-bot이 legal move를 생성못하면 사용자의 체스 정보를 삭제 및 사용자가 승리했다는 문구 표시 (Discord)
    if not move_success:
        del user_chess[interaction.user.id]
        del user_count[interaction.user.id]
        del user_prompt[interaction.user.id]
        await interaction.followup.send("<@" + str(interaction.user.id) + "> won the game.", ephemeral=True)
        return

    legal_move = str(user_chess[interaction.user.id].legal_moves)
    legal_move = legal_move[legal_move.index("(") + 1 : legal_move.index(")")].split(", ")

    # chess-bot의 move 이후에 legal move가 없으면 사용자의 체스 정보를 삭제 및 사용자가 패배했다는 문구 표시 (Discord)
    if not legal_move:
        del user_chess[interaction.user.id]
        del user_count[interaction.user.id]
        del user_prompt[interaction.user.id]
        await interaction.followup.send("<@" + str(interaction.user.id) + "> lost the game.", ephemeral=True)
        return

    embed = discord.Embed(
        title="Your game with Chess Bot",
        description="Your opponent has moved: **" + move[0] + "**",
        color=discord.Color.green(),
    )
    embed.set_footer(text="Legal Move: " + ", ".join(legal_move))
    button = Button(label="Resign", style=discord.ButtonStyle.gray, emoji="🏳️")

    # 버튼을 클릭시, 사용자의 체스 정보를 삭제 및 사용자가 포기했다는 문구 표시 (Discord)
    async def button_callback(interaction):
        del user_chess[interaction.user.id]
        del user_count[interaction.user.id]
        del user_prompt[interaction.user.id]
        await interaction.response.send_message(
            "<@" + str(interaction.user.id) + "> resigned the game.", ephemeral=True
        )

    button.callback = button_callback

    view = View()
    view.add_item(button)
    await interaction.followup.send(
        embed=embed, file=display_board(user_chess[interaction.user.id]), view=view, ephemeral=True
    )  # chess-bot의 move 및 버튼 표시 (Discord)


user_chess = {}  # {'user_id': chess.Board(), ...}
user_color = {}  # {'user_id': 'White', ...}
user_count = defaultdict(int)  # {'user_id': 1, ...}
user_prompt = defaultdict(list)  # {'user_id': ['1.', 'e4', 'e5'], ...}


@tree.command(
    name="start", description="Start a playing chess game.", guild=discord.Object(id=int(os.environ.get("GUILD_ID")))
)
@app_commands.choices(
    color=[app_commands.Choice(name="White", value="White"), app_commands.Choice(name="Black", value="Black")]
)
async def start(interaction: discord.Interaction, color: app_commands.Choice[str]):

    if color.value == "White":
        user_chess[interaction.user.id] = chess.Board()
        user_color[interaction.user.id] = "White"
        legal_move = str(user_chess[interaction.user.id].legal_moves)
        legal_move = legal_move[legal_move.index("(") + 1 : legal_move.index(")")].split(", ")

        embed = discord.Embed(
            title="Game Started!",
            description="Black: Chess Bot\nWhite: `"
            + str(interaction.user.name)
            + "`\n\nUse `/move` to move your piece.",
            color=discord.Color.blue(),
        )
        embed.set_footer(text="Legal Move: " + ", ".join(legal_move))
        await interaction.response.send_message(
            embed=embed, file=display_board(user_chess[interaction.user.id]), ephemeral=True
        )  # 체스판 초기상황 표기 (Discord)

    elif color.value == "Black":
        user_color[interaction.user.id] = "Black"
        user_chess[interaction.user.id] = chess.Board()
        legal_move = str(user_chess[interaction.user.id].legal_moves)
        legal_move = legal_move[legal_move.index("(") + 1 : legal_move.index(")")].split(", ")

        embed = discord.Embed(
            title="Game Started!",
            description="Black: `"
            + str(interaction.user.name)
            + "`\nWhite: Chess Bot\n\nUse `/move` to move your piece.",
            color=discord.Color.blue(),
        )
        file = display_board(user_chess[interaction.user.id])

        user_count[interaction.user.id] += 1
        user_prompt[interaction.user.id].append(str(user_count[interaction.user.id]) + ".")
        await bot_move(interaction, embed, file)  # 체스판 초기상황 표기 및 chess-bot 움직임 표기 (Discord)


@tree.command(
    name="move",
    description="Move your piece among legal moves.",
    guild=discord.Object(id=int(os.environ.get("GUILD_ID"))),
)
async def move(interaction: discord.Interaction, move: str):
    # 사용자가 게임을 시작 안했거나 게임이 끝나, 체스판 정보에 없을 때에 /move 시도 시 에러 표기 (Discord)
    if interaction.user.id not in user_chess.keys():
        embed = discord.Embed(
            title="You do not have a game in progress.",
            description="Use `/start` to start a game.",
            color=discord.Color.red(),
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return

    legal_move = str(user_chess[interaction.user.id].legal_moves)
    legal_move = legal_move[legal_move.index("(") + 1 : legal_move.index(")")].split(", ")
    # /move를 이용해 illegal move 시도 시 에러 표기 (Discord)
    if move not in legal_move:
        embed = discord.Embed(
            title="Illegal Move Played.", description="Legal Move: " + ", ".join(legal_move), color=discord.Color.red()
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return

    if user_color[interaction.user.id] == "White":
        user_count[interaction.user.id] += 1
        user_prompt[interaction.user.id].append(str(user_count[interaction.user.id]) + ". ")  # 1.e4 e5 -> 1. e4 e5 2.
        embed, file = await human_move(interaction, move)  # 1. e4 e5 2. -> 1. e4 e5 2. d4
        await bot_move(interaction, embed, file)  # 1. e4 e5 2. d4 -> 1. e4 e5 2. d4 d5

    elif user_color[interaction.user.id] == "Black":
        embed, file = await human_move(interaction, move)  # 1. e4 -> 1. e4 e5
        user_count[interaction.user.id] += 1
        user_prompt[interaction.user.id].append(str(user_count[interaction.user.id]) + ".")  # 1. e4 e5 -> 1. e4 e5 2.
        await bot_move(interaction, embed, file)  # 1. e4 e5 2. -> 1. e4 e5 2. d4


client.run(os.environ.get("TOKEN"))
