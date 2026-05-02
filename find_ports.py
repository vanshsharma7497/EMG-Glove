import serial.tools.list_ports

def list_available_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("  [X] No COM ports found on this computer.")
        return

    print("\n  Available COM Ports:")
    print("  " + "─" * 60)
    for p in ports:
        print(f"  Port: {p.device:<10} | Description: {p.description}")
        # print(f"        HWID: {p.hwid}")
    print("  " + "─" * 60)

if __name__ == '__main__':
    list_available_ports()


