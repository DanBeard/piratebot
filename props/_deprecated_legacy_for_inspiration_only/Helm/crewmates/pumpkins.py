import asyncio
import json

import requests

from crewmate import BaseCrewmate, CrewmateProperty
from util import MessageTypes, MessageFields
import os
import random
import xmltodict

# raspberry pi imports
dir_path = os.path.dirname(os.path.realpath(__file__))
on_pi = None
LIGHT_PIN = 5
SING_TIMEOUT_SEC = 60*10

try:
    import RPi.GPIO as GPIO
    on_pi = True
    VLC_HOST = "127.0.0.1:8080"
except ModuleNotFoundError as e:
    on_pi = False
    VLC_HOST = "192.168.1.104:8080"

VIDEOS = [  # name of video and length in secs
    "eliza_lee.mp4",
    "old_maui.mp4",
    "randy.mp4",
    "fire_marengo.mp4",
    "rosibella.mp4",
    "joli_rouge.mp4"
]

IDLE_VIDEO = "idle2.mp4"

VIDEO_DIR = "/home/pi/videos/"

class Pumpkins(BaseCrewmate):
    """
    Singing Pumpkins
    """

    song = CrewmateProperty()
    singing = CrewmateProperty()
    song_list = CrewmateProperty()

    def __init__(self):
        self.address = "PUMPKINS"
        self.singing = False
        self.song = None
        self.song_list = VIDEOS
        self.last_song_idx = 0

        self._blocking = BlockingPumpkins(VLC_HOST)
        self._is_fullscreen = True
        self._idle_id = 0
        self._song_info = {}
        self._last_song_command_num = 0  # keep a counter of singing commands so we don't have async issues

        if on_pi:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(LIGHT_PIN, GPIO.OUT)

        super(Pumpkins, self).__init__()

    async def on_connection(self, ws):
        await self.load_videos()
        await self.go_idle()
        asyncio.get_event_loop().create_task(self.singloop())
        print("READY")

    async def handle_command(self, msg):
        print(msg)
        if msg[MessageFields.DATA]["command"] == "SING":
            # sing what was asked, or randomly choose oen
            song_name = msg[MessageFields.DATA].get("song_name", VIDEOS[self.last_song_idx % len(VIDEOS)])
            self.last_song_idx +=1
            back_to_idle = await self.sing(song_name)

        elif msg[MessageFields.DATA]["command"] == "GO_IDLE":
            await self.go_idle()

    async def load_videos(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._blocking.clear_playlist)
        await asyncio.sleep(0.5)
        for x in [IDLE_VIDEO] + VIDEOS:
            await loop.run_in_executor(None, self._blocking.add_to_playlist, VIDEO_DIR+x)
            await asyncio.sleep(0.5)
        playlist_root = await loop.run_in_executor(None, self._blocking.get_playlist)
        playlist_leafs = playlist_root["node"]["node"][0]["leaf"]
        for p in playlist_leafs:
            name, pid, duration = p["@name"], p["@id"], p["@duration"]
            self._song_info[name] = {"name": name, "id": pid, "duration": duration}
            if name == IDLE_VIDEO:
                self._idle_id = pid

        print(json.dumps(playlist_root["node"]["node"][0]["leaf"]))

    async def sing(self, song_name):
        print("singing..."+song_name)
        self._last_song_command_num += 1
        my_command_num = self._last_song_command_num

        loop = asyncio.get_event_loop()
        song_info = self._song_info[song_name]

        if on_pi:
            GPIO.output(LIGHT_PIN, 1)

        self.singing = True
        self.song = song_name
        await loop.run_in_executor(None, self._blocking.play_by_id, song_info["id"])
        await asyncio.sleep(int(song_info["duration"]))
        if my_command_num == self._last_song_command_num:
            await self.go_idle()
            return True
        return False

    async def go_idle(self):
        self.singing = False
        self.song = None
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._blocking.play_by_id, self._idle_id)
        if on_pi:
            GPIO.output(LIGHT_PIN, 0)

    async def singloop(self):
        await asyncio.sleep(5)
        print("Starting singloop")
        self.last_song_idx = 0
        while True:
            song_name = VIDEOS[self.last_song_idx % len(VIDEOS)]
            self.last_song_idx += 1
            if not self.singing:
                await self.sing(song_name)
            await asyncio.sleep(SING_TIMEOUT_SEC)


class BlockingPumpkins:
    """
    A list of blocking requests calls that must be run in a thread pool
    """

    def __init__(self, host):
        self.host = host
        self.base_uri = f"http://{host}/requests/"
        self._auth = ("", "x")

    def add_to_playlist(self, filename):
        resp = requests.get(f"{self.base_uri}status.xml?command=in_play&input={filename}&fullscreen=1", auth=self._auth)
        return xmltodict.parse(resp.content)

    def clear_playlist(self):
        resp = requests.get(f"{self.base_uri}status.xml?command=pl_empty", auth=self._auth)
        return xmltodict.parse(resp.content)

    def get_playlist(self):
        resp = requests.get(f"{self.base_uri}playlist.xml", auth=self._auth)
        return xmltodict.parse(resp.content)

    def play_by_id(self, id):
        resp = requests.get(f"{self.base_uri}status.xml?command=pl_play&id={id}", auth=self._auth)
        print(f"{self.base_uri}status.xml?command=pl_play&id={id}")
        return xmltodict.parse(resp.content)


if __name__ == "__main__":
    qm = Pumpkins()
    asyncio.get_event_loop().run_until_complete(qm.start())
    asyncio.get_event_loop().run_forever()
