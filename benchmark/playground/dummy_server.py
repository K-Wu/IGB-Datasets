# From https://discuss.dgl.ai/t/two-node-job-failure-during-network-configurations/4231/4?u=k-wu
import socket

def start_dummy_server(host='127.0.0.1', port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Listening on {host}:{port}...")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            conn.sendall(b'Hello, client')

if __name__ == '__main__':
    import sys
    print("sys.argv: ", sys.argv)
    print("Using ip: ", sys.argv[1])
    start_dummy_server(host = sys.argv[1])