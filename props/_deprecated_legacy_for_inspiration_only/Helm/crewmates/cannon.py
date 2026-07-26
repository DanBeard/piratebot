import asyncio
from crewmate import BaseCrewmate, CrewmateProperty
from util import MessageTypes, MessageFields
from playsound import playsound
import os

# raspberry pi imports
dir_path = os.path.dirname(os.path.realpath(__file__))
on_pi = None
LIGHT_PIN = 5
try:
    import RPi.GPIO as GPIO
    on_pi = True
except ModuleNotFoundError as e:
    on_pi = False


class Cannon(BaseCrewmate):
    """
    The cannon.... fires the cannon
    """

    firing = CrewmateProperty()

    def __init__(self):
        self.address = "CANNON"
        self.firing = False

        if on_pi:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(LIGHT_PIN, GPIO.OUT)

        super(Cannon, self).__init__()

    async def handle_command(self, msg):
        if msg[MessageFields.DATA]["command"] == "FIRE":
            if self.firing:
                return  # TODO: an audio queue when firing too fast
            try:
                print("**BOOM**")

                if on_pi:
                    GPIO.output(LIGHT_PIN, 1)
                self.firing = True
                await asyncio.sleep(0.5)
                playsound(dir_path+"/sounds/nri-cannon.mp3", block=False)
                await asyncio.sleep(3.5)
                if on_pi:
                    GPIO.output(LIGHT_PIN, 0)

                # give the cannon a rest
                await asyncio.sleep(2.5)
                # await asyncio.sleep(0.5)
                # playsound(dir_path+"/sounds/splash.mp3", block=False)
                # await asyncio.sleep(2.5)

            finally:
                self.firing = False


if __name__ == "__main__":
    qm = Cannon()
    asyncio.get_event_loop().run_until_complete(qm.start())
    asyncio.get_event_loop().run_forever()
