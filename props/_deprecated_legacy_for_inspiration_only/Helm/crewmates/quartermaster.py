import asyncio
from crewmates.crewmate import BaseCrewmate, CrewmateProperty
from crewmates.pumpkins import VIDEOS as PUMPKIN_VIDEOS


class QM_STATES:
    INIT = "init"  # Starting up
    IDLE = "idle"  # Basic idle state & audio ambiance
    QUIET = "quiet"  # remove audio ambiance (e.g. for singing pumpkins)



class QuarterMaster(BaseCrewmate):
    """
    The Quartermaster handles crewmate syncing and the state of the ambience in general.
    If an action involves coordinating between many crewmates, then the logic/command belongs here
    """

    state = CrewmateProperty()
    
    def __init__(self):
        self.address = "QUARTERMASTER"
        self.state = QM_STATES.INIT
        self._last_song_idx = 0
        super(QuarterMaster, self).__init__()

    async def on_connection(self, ws):
        asyncio.create_task(self.do_init())

    async def do_init(self):
        await asyncio.sleep(5)
        self.state = QM_STATES.IDLE
        await asyncio.sleep(5)
        await self.sing()

    # Time for the pumpkins to sing!
    # async def sing_next(self):
    #     song_name = PUMPKIN_VIDEOS[self._last_song_idx % len(PUMPKIN_VIDEOS)]
    #     self._last_song_idx += 1
    #     print("SINGING..." + song_name)
    #     self.state = QM_STATES.QUIET
    #     await self.command("PUMPKINS", {"command": "SING", "song_name": song_name})

if __name__ == "__main__":
    qm = QuarterMaster()
    asyncio.get_event_loop().run_until_complete(qm.start())
    asyncio.get_event_loop().run_forever()
