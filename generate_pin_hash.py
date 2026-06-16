import base64
import getpass
import hashlib
import secrets


HASH_PREFIX = "pbkdf2_sha256"
ITERATIONS = 310_000


def main() -> None:
    pin = getpass.getpass("Enter 4-digit PIN: ").strip()
    confirm = getpass.getpass("Confirm 4-digit PIN: ").strip()

    if pin != confirm:
        raise SystemExit("PIN values do not match.")

    if not pin.isdigit() or len(pin) != 4:
        raise SystemExit("PIN must be exactly 4 digits.")

    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, ITERATIONS)
    encoded_salt = base64.b64encode(salt).decode("ascii")
    encoded_digest = base64.b64encode(digest).decode("ascii")

    print(f"{HASH_PREFIX}${ITERATIONS}${encoded_salt}${encoded_digest}")


if __name__ == "__main__":
    main()
