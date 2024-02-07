import socket

def connect_to_dummy_server(host='127.0.0.1', port=30050):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"Connected to {host}:{port}")
        data = s.recv(1024)
        print(f"Received: {data.decode()}")

if __name__ == '__main__':
    import sys
    print("sys.argv: ", sys.argv)
    print("Using ip: ", sys.argv[1])
    connect_to_dummy_server(host = sys.argv[1])
