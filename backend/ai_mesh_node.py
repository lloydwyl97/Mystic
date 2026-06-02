import logging
import socket

logger = logging.getLogger(__name__)


def start_node(host="localhost", port=9444):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # allow quick restart
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
        s.listen(5)
        logger.info(f"[MESH] Listening on {host}:{port}")
        while True:
            try:
                conn, addr = s.accept()
            except KeyboardInterrupt:
                logger.info("[MESH] Shutting down")
                break
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.exception("[MESH] Error accepting connection")
                continue

            # ensure the connection is always closed
            try:
                try:
                    data = conn.recv(1024)
                    if not data:
                        logger.info(f"[MESH] Connection from {addr} closed without data")
                        continue
                    try:
                        text = data.decode()
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        text = data.decode(errors="replace")
                    logger.info(f"[MESH] Received from {addr}: {text}")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    logger.exception(f"[MESH] Error handling connection from {addr}")
                    continue
            finally:
                try:
                    conn.close()
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    logger.exception(f"[MESH] Error closing connection from {addr}")
    finally:
        try:
            s.close()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("[MESH] Error closing socket")
