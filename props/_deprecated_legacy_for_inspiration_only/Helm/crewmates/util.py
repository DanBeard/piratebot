import asyncio
import socket
import struct
from asyncio import DatagramProtocol, transports
from typing import Tuple


class MessageTypes:
    COMMAND = "C"
    NOTIFY = "N"
    SUBSCRIBE = "S"
    LOG = "L"
    GET_MANIFEST = "GM"
    MANIFEST = "M"
    SET_ADDRESS = "SA"

class MessageFields:
    TYPE = "type"
    ADDRESS = "address"  # for a command this is the TO field. For a  Notify this is a FROM field.
    DATA = "data"
    FROM = "from"


# UDP multicast
MCAST_GRP = '224.1.33.7'
MCAST_PORT = 31338

class CrewShout(DatagramProtocol):

    def __init__(self, is_server):
        self.is_server = is_server
        self.transport = None
        self.servers_found = []

    def connection_made(self, transport: transports.BaseTransport) -> None:
        self.transport = transport

        # Allow receiving multicast broadcasts
        sock = self.transport.get_extra_info('socket')
        group = socket.inet_aton(MCAST_GRP)
        mreq = struct.pack('4sL', group, socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        print(f"{data.decode()} ->  {addr[0]}")
        # new crewmate greeting. Welcome them aboard
        if self.is_server and data == b"AHOY":
            self.transport.sendto(b"WELCOME ABOARD", addr) # direct to them
        if not self.is_server and data == b"WELCOME ABOARD":
            self.servers_found.append(addr[0])

    def shout(self):
        self.transport.sendto(b"AHOY", (MCAST_GRP, MCAST_PORT))

    def server_shout(self):
        # Warning shouted by a ship. If we hear this after launch, there's two decks that have launched!
        self.transport.sendto(b"AVAST! NEW SHIP IN HARBOR", (MCAST_GRP, MCAST_PORT))


async def shout_server():
    _, deck_shouter = await asyncio.get_event_loop().create_datagram_endpoint(lambda: CrewShout(True), ("0.0.0.0", MCAST_PORT))
    deck_shouter.server_shout()


async def shout_client():
    port_offset = 0
    succ = False
    # try different ports in case there's multiple clients on one host
    while not succ and port_offset < 100:
        try:
            shout = CrewShout(False)
            _, deck_shouter = await asyncio.get_event_loop().create_datagram_endpoint(lambda: shout, ("0.0.0.0", MCAST_PORT+port_offset))
            while len(shout.servers_found) == 0:
                print("~LFC~")
                deck_shouter.shout()
                await asyncio.sleep(1)
            return shout.servers_found[0]
        except OSError as e:
            print(f"{MCAST_PORT+port_offset} already in use. Trying the next one")
            port_offset = port_offset + 1

