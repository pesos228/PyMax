from uuid import UUID, uuid4

from pymax import SocketMaxClient


PHONE = "+79991234567"


def test_client_can_skip_sqlite_session_persistence(tmp_path):
    device_id = uuid4()
    work_dir = tmp_path / "sessionless"

    client = SocketMaxClient(
        phone=PHONE,
        token="postgres-token",
        device_id=device_id,
        work_dir=str(work_dir),
        persist_session=False,
    )

    assert client.auth_token == "postgres-token"
    assert client.device_id == device_id
    assert client.persist_session is False
    assert client._database is None
    assert not (work_dir / "session.db").exists()


def test_explicit_token_takes_precedence_over_sqlite_session(tmp_path):
    first_device_id = uuid4()
    work_dir = tmp_path / "session"
    SocketMaxClient(
        phone=PHONE,
        token="old-token",
        device_id=first_device_id,
        work_dir=str(work_dir),
    )

    second_device_id = uuid4()
    client = SocketMaxClient(
        phone=PHONE,
        token="new-token",
        device_id=second_device_id,
        work_dir=str(work_dir),
    )

    assert client.auth_token == "new-token"
    assert client.device_id == second_device_id
    assert client._database is not None
    assert client._database.get_auth_token() == "new-token"


def test_export_session_returns_adapter_friendly_values(tmp_path):
    device_id = uuid4()
    client = SocketMaxClient(
        phone=PHONE,
        token="token-for-postgres",
        device_id=device_id,
        work_dir=str(tmp_path),
        persist_session=False,
    )

    exported = client.export_session()

    assert exported == {
        "phone": PHONE,
        "token": "token-for-postgres",
        "device_id": str(device_id),
    }
    assert UUID(exported["device_id"]) == device_id
