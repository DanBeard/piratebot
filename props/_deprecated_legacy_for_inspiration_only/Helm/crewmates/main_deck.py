import asyncio
from collections import defaultdict

import websockets
from websockets.exceptions import ConnectionClosed
import json
from util import MessageTypes, MessageFields, shout_server


class MainDeck:
    """
    The Main Deck handles routing the messages to and from the crewmates (controllers).
    It also keeps a manifest of all connected crewmates
    It also serves the UI over HTTP
    """

    def __init__(self):
        self.subs = defaultdict(list)
        self.manifest = []

    async def start(self, ws, path):
        print(str(ws.remote_address) + " Permission to board?")
        my_address = None
        try:
            async for rawmsg in ws:
                msg = json.loads(rawmsg)
                print(msg)

                # handling here
                msg_type = msg[MessageFields.TYPE]
                if msg_type == MessageTypes.SUBSCRIBE:
                    self.subs[msg[MessageFields.ADDRESS]].append(ws)
                elif msg_type == MessageTypes.SET_ADDRESS:
                    my_address = msg[MessageFields.ADDRESS]
                    self.manifest.append(my_address)
                    await self.beacon_manifest()
                elif msg_type == MessageTypes.GET_MANIFEST:
                    await self.send_manifest(ws, my_address)
                else:
                    for listener in self.subs[msg[MessageFields.ADDRESS]] + self.subs["*"]:
                        if listener != ws:  # don't echo back the message to the sender
                            await listener.send(rawmsg)
        finally:
            print(f"{ws.remote_address} aka {my_address} has left the ship ")
            if my_address in self.manifest:
                self.manifest.remove(my_address)

            # remove from any listeners
            for v in self.subs.values():
                if ws in v:
                    v.remove(ws)

            # beacon the manifest for anyone listening
            await self.beacon_manifest()

    async def send_manifest(self, ws, to_address):
        await ws.send(json.dumps({
            MessageFields.TYPE: MessageTypes.MANIFEST,
            MessageFields.ADDRESS: to_address,
            MessageFields.DATA: self.manifest
        }))

    async def beacon_manifest(self):
        for listener in self.subs["*"]:
            await self.send_manifest(listener, "*")




if __name__ == "__main__":
    md = MainDeck()
    start_server = websockets.serve(md.start, "0.0.0.0", 31337)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_until_complete(shout_server())
    loop.run_forever()

