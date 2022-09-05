import io
import os
import random
from collections import defaultdict
from typing import List

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
seed = random.randint(1, 2**31 - 1)
torch.manual_seed(seed)
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>"
)
config = GPT2Config.from_pretrained(os.environ.get("MODEL_PATH", "./model"), output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained(os.environ.get("MODEL_PATH", "./model"), config=config)
model.resize_token_embeddings(len(tokenizer))


# 디스코드 봇 로그인
class DiscordClient(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.synced = False

    async def on_ready(self):
        await self.wait_until_ready()
        if not self.synced:
            await tree.sync(guild=discord.Object(id=int(os.environ.get("GUILD_ID"))))
            self.synced = True
        print(f"We have logged in as {self.user}.")


client = DiscordClient()
tree = app_commands.CommandTree(client)


# 모델을 이용해 prompt의 다음 토큰 생성
def get_next_moves(prompt: List[str]) -> List[str]:
    # prompt 이후 5번 Text Generation
    # e.g) 1. e4 -> 1. e4 e5 ...
    prompt = " ".join(prompt)
    input_ids = tokenizer.encode(f"<|startoftext|>{prompt}", return_tensors="pt")
    sample_outputs = model.generate(
        input_ids,
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
    moves = tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
    next_move_count = defaultdict(int)
    for move in moves:
        move = move[len(prompt) + 1 :]
        next_move_count[move[: move.index(" ")]] += 1

    next_moves = sorted(next_move_count.items(), key=lambda x: -x[1])
    next_moves = [move[0] for move in next_moves]

    return next_moves


def get_legal_moves(user_id: str) -> List[str]:
    legal_moves = list(user_chess[user_id].legal_moves)
    legal_moves = [user_chess[user_id].san(move) for move in legal_moves]

    return legal_moves


def increase_turn_count(user_id: str):
    user_count[user_id] += 1
    user_prompt[user_id].append(str(user_count[user_id]) + ".")


def delete_user_chess_info(user_id: str):
    del user_chess[user_id]
    del user_count[user_id]
    del user_prompt[user_id]


# 체스판을 시각화
def display_board(chess_board: chess.Board) -> discord.File:
    with io.BytesIO() as binary:
        board = Generator.generate(chess_board).resize((500, 500), Image.Resampling.LANCZOS)
        board.save(binary, "PNG")
        binary.seek(0)
        file = discord.File(fp=binary, filename="board.png")

    return file


async def human_move(interaction: discord.Interaction, move: str):
    user_id = interaction.user.id
    user_name = interaction.user.name

    # 사용자가 /move로 입력한 move를 추가
    user_chess[user_id].push_san(move)
    user_prompt[user_id].append(move)

    embed = discord.Embed(
        title="Your game with Chess Bot",
        description=f"`{user_name}` have moved: **{move}**",
        color=discord.Color.green(),
    )
    view = View()
    await interaction.response.send_message(
        embed=embed, file=display_board(user_chess[user_id]), view=view, ephemeral=True
    )


async def bot_move(interaction: discord.Interaction):
    user_id = interaction.user.id

    # await interaction.response.defer(ephemeral=True)  # response 이전까지 chess-bot is thinking... 문구 표시 (Discord)
    next_moves = get_next_moves(user_prompt[user_id])
    legal_moves = get_legal_moves(user_id)

    # 생성한 chess-bot의 move를 추가
    move_success = False
    for move in next_moves:
        if move in legal_moves:
            user_chess[user_id].push_san(move)
            user_prompt[user_id].append(move)
            move_success = True
            break

    # chess-bot이 legal move를 생성못하면 사용자의 체스 정보를 삭제 및 사용자가 승리했다는 문구 표시 (Discord)
    if not move_success:
        delete_user_chess_info(user_id)
        await interaction.followup.send(f"<@{user_id}> won the game.", ephemeral=True)
        return

    legal_moves = get_legal_moves(user_id)

    # chess-bot의 move 이후에 legal move가 없으면 사용자의 체스 정보를 삭제 및 사용자가 패배했다는 문구 표시 (Discord)
    if not legal_moves:
        delete_user_chess_info(user_id)
        await interaction.followup.send(f"<@{user_id}> lost the game.", ephemeral=True)
        return

    embed = discord.Embed(
        title="Your game with Chess Bot",
        description=f"Your opponent has moved: **{move}**",
        color=discord.Color.green(),
    )
    embed.set_footer(text=f'Legal Move: {", ".join(legal_moves)}')
    button = Button(label="Resign", style=discord.ButtonStyle.gray, emoji="🏳️")

    # 버튼을 클릭시, 사용자의 체스 정보를 삭제 및 사용자가 포기했다는 문구 표시 (Discord)
    async def button_callback(interaction: discord.Interaction):
        delete_user_chess_info(user_id)
        await interaction.response.send_message(f"<@{user_id}> resigned the game.", ephemeral=True)

    button.callback = button_callback

    view = View()
    view.add_item(button)
    await interaction.followup.send(
        embed=embed, file=display_board(user_chess[user_id]), view=view, ephemeral=True
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
    user_id = interaction.user.id
    user_name = interaction.user.name

    color = color.value
    user_chess[user_id] = chess.Board()
    user_color[user_id] = color

    if color == "White":
        legal_moves = get_legal_moves(user_id)

        embed = discord.Embed(
            title="Game Started!",
            description=f"Black: Chess Bot\nWhite: `{user_name}`\n\nUse `/move` to move your piece.",
            color=discord.Color.blue(),
        )
        embed.set_footer(text=f'Legal Move: {", ".join(legal_moves)}')
        await interaction.response.send_message(
            embed=embed, file=display_board(user_chess[user_id]), ephemeral=True
        )  # 체스판 초기상황 표기 (Discord)

    elif color == "Black":
        embed = discord.Embed(
            title="Game Started!",
            description=f"Black: `{user_name}`\nWhite: Chess Bot\n\nUse `/move` to move your piece.",
            color=discord.Color.blue(),
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

        increase_turn_count(user_id)
        await bot_move(interaction)  # 체스판 초기상황 표기 및 chess-bot 움직임 표기 (Discord)


@tree.command(
    name="move",
    description="Move your piece among legal moves.",
    guild=discord.Object(id=int(os.environ.get("GUILD_ID"))),
)
async def move(interaction: discord.Interaction, move: str):
    user_id = interaction.user.id

    # 사용자가 게임을 시작 안했거나 게임이 끝나, 체스판 정보에 없을 때에 /move 시도 시 에러 표기 (Discord)
    if user_id not in user_chess.keys():
        embed = discord.Embed(
            title="You do not have a game in progress.",
            description="Use `/start` to start a game.",
            color=discord.Color.red(),
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return

    legal_moves = get_legal_moves(user_id)

    if move not in legal_moves:
        embed = discord.Embed(
            title="Illegal Move Played.",
            description=f'Legal Move: {", ".join(legal_moves)}',
            color=discord.Color.red(),
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return

    if user_color[user_id] == "White":
        increase_turn_count(user_id)  # 1.e4 e5 -> 1. e4 e5 2.
        await human_move(interaction, move)  # 1. e4 e5 2. -> 1. e4 e5 2. d4
        await bot_move(interaction)  # 1. e4 e5 2. d4 -> 1. e4 e5 2. d4 d5

    elif user_color[user_id] == "Black":
        await human_move(interaction, move)  # 1. e4 -> 1. e4 e5
        increase_turn_count(user_id)  # 1. e4 e5 -> 1. e4 e5 2.
        await bot_move(interaction)  # 1. e4 e5 2. -> 1. e4 e5 2. d4


client.run(os.environ.get("TOKEN"))
