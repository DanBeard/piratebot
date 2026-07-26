import asyncio
from crewmate import BaseCrewmate, CrewmateProperty
from util import MessageTypes, MessageFields
from playsound import playsound
import os
import subprocess
import alsaaudio

# raspberry pi imports
dir_path = os.path.dirname(os.path.realpath(__file__))
on_pi = None
LIGHT_PIN = 5
try:
    import RPi.GPIO as GPIO
    on_pi = True
except ModuleNotFoundError as e:
    on_pi = False

VOICE_LINE_TIMEOUT = 60

class Ambiance(BaseCrewmate):
    """
    Pirate ambiance. Spooking music and pirate phrases.(Probably hidden in the captain's ship)
    TODO: Also laser and strobe light control for lightening?
    """

    def __init__(self):
        self.address = "AMBIANCE"
        self.volume = -1

        if on_pi:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(LIGHT_PIN, GPIO.OUT)

        super(Ambiance, self).__init__()

    async def on_connection(self, ws):
        # play the music in VLC on repeat
        subprocess.Popen(["vlc", dir_path+"/sounds/ambiance/music", "--loop"])
        # await asyncio.sleep(5)
        asyncio.get_event_loop().create_task(self.say_voicelines())
        await self.set_volume(100)

        await self.on_prop_change("PUMPKINS", "singing", self.on_pumpkin_singing_change)

    async def on_pumpkin_singing_change(self, val):
        # we go quiet when pumpkins are singing
        print("Changing volume " + str(val))
        await self.set_volume(0 if val else 100)

    async def say_voicelines(self):
        await asyncio.sleep(3)
        print("Saying Voicelines")
        while True:
            for f in os.listdir(dir_path+"/sounds/ambiance/voice_lines/"):
                if self.volume > 0:
                    # TODO: set GPIO to lightup skele (don't forget to sjp if volume is 0)
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.playsound_blocking, dir_path+"/sounds/ambiance/voice_lines/"+f)
                else:
                    print("Skipping vol="+str(self.volume))
                await asyncio.sleep(VOICE_LINE_TIMEOUT)

    def playsound_blocking(self, path):
        print("saying " + path)
        if on_pi:
            GPIO.output(LIGHT_PIN, 1)
        playsound(path, block=True)
        if on_pi:
            GPIO.output(LIGHT_PIN, 0)
        print("finished" + path)

    async def handle_command(self, msg):
        if msg[MessageFields.DATA].get("command", "") == "VOLUME":
            await self.set_volume(msg[MessageFields.DATA].get("vol", 50))

    async def set_volume(self, vol):
        m = alsaaudio.Mixer()
        new_vol = self.volume
        # fade
        while self.volume != int(vol):
            diff = min(abs(vol-self.volume), 5)
            new_vol += (diff if vol > self.volume else -1*diff)
            new_vol = max(new_vol, 0)
            print(new_vol)
            m.setvolume(new_vol)
            self.volume = new_vol
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    qm = Ambiance()
    asyncio.get_event_loop().run_until_complete(qm.start())
    asyncio.get_event_loop().run_forever()
