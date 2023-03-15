import threading
import sys
import socket
import time
import json
import os
import re
import struct
import signal
from datetime import datetime
import hivemind


def send():
    MCAST_GRP = "224.1.1.1"
    MCAST_PORT = (
        5008  # feature request mit anderen port oder anderen IP für andere Wahl, SPÄTER
    )
    MULTICAST_TTL = 2
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
    while True:
        now = datetime.now()
        data = {}
        data["ips"] = []
        for addr in dht.get_visible_maddrs():
            if "127.0.0.1" not in str(addr):
                data["ips"].append(str(addr))
        data["node"] = re.findall(r"\d+", os.uname()[1])[0]
        data["time"] = str(now)
        json_data = json.dumps(data)
        sock.sendto(json_data.encode("utf-8"), (MCAST_GRP, MCAST_PORT))
        time.sleep(6)


def receive():  # receiveTest.py for references
    MCAST_GRP = "224.1.1.1"
    MCAST_PORT = 5008
    IS_ALL_GROUPS = True
    vm_dict = {}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if IS_ALL_GROUPS:
        sock.bind(("", MCAST_PORT))
    else:
        sock.bind((MCAST_GRP, MCAST_PORT))
    mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    while True:
        # For Python 3, change next line to "print(sock.recv(10240))"
        jsonString = sock.recv(10240).decode()
        json_object = json.loads(jsonString)
        tempNode = json_object["node"]
        tempList = [json_object["ips"], json_object["time"]]
        vm_dict.update({tempNode: tempList})
        print(vm_dict)
        with open("tmpFile.json", "w") as f:
            f.write(str(vm_dict))
            f.flush()
            os.fsync(f.fileno())
            os.replace("tmpFile.json", "hiveTest.json")


if __name__ == "__main__":
    try:
        dht = hivemind.DHT(
            host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"], start=True
        )  # wie connnecten mit den anderen Teil von Hivemind?
        send = threading.Thread(target=send)
        receive = threading.Thread(target=receive)
        send.start()
        receive.start()
    except KeyboardInterrupt or signal.SIGTERM:
        print("Ctrl+C pressed...")
        while send.is_alive():
            send.join(1)  # time out not to block KeyboardInterrupt
        while receive.is_alive():
            receive.join(1)  # time out not to block KeyboardInterrupt
        sys.exit(1)
