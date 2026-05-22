import asyncio
import logging
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pymax.files import Audio
from pymax.mixins.message import MessageMixin
from pymax.payloads import AttachAudioPayload
from pymax.static.enum import AttachType


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg is required for audio conversion tests",
)


def _run(coro):
    return asyncio.run(coro)


def _make_wav(path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.8",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _make_ogg(path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.8",
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            "-application",
            "voip",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_audio_converts_wav_to_ogg_opus(tmp_path):
    source = tmp_path / "source.wav"
    _make_wav(source)

    audio = Audio(path=str(source))
    prepared = _run(audio.prepare())

    assert prepared.path.read_bytes().startswith(b"OggS")
    assert b"OpusHead" in prepared.path.read_bytes()[:4096]
    assert 700 <= prepared.duration_ms <= 900
    assert len(prepared.wave) == 80
    assert min(prepared.wave) >= 1
    assert max(prepared.wave) <= 100

    output = prepared.path
    audio.cleanup()
    assert not output.exists()


def test_audio_accepts_ready_ogg_without_removing_source(tmp_path):
    source = tmp_path / "voice.ogg"
    _make_ogg(source)

    audio = Audio(path=str(source), convert=False)
    prepared = _run(audio.prepare())

    assert prepared.path == source
    assert 700 <= prepared.duration_ms <= 900
    assert len(prepared.wave) == 80
    assert min(prepared.wave) >= 1
    assert max(prepared.wave) <= 100

    audio.cleanup()
    assert source.exists()


def test_attach_audio_payload_shape():
    payload = AttachAudioPayload(
        duration=1234,
        wave=b"\x01" * 80,
        token="token",
    ).model_dump(by_alias=True)

    assert payload["_type"] == AttachType.AUDIO
    assert payload["duration"] == 1234
    assert payload["wave"] == b"\x01" * 80
    assert payload["token"] == "token"


def test_audio_send_retry_on_attachment_not_ready():
    class FakeClient(MessageMixin):
        AUDIO_SEND_RETRY_DELAY = 0

        def __init__(self):
            self.calls = 0
            self.logger = logging.getLogger("test")

        async def _send_and_wait(self, opcode, payload, cmd=0, timeout=20.0):
            self.calls += 1
            if self.calls == 1:
                return {"payload": {"error": "attachment.not.ready"}}
            return {
                "payload": {
                    "chatId": 1,
                    "message": {
                        "id": 10,
                        "time": 20,
                        "text": "ok",
                        "type": "TEXT",
                        "sender": 30,
                    },
                },
            }

        async def _get_chat(self, chat_id):
            return None

        async def _queue_message(self, *args, **kwargs):
            return None

        def _create_safe_task(self, coro, name=None):
            return asyncio.create_task(coro, name=name)

    client = FakeClient()
    message = _run(client._send_message_payload({"chatId": 1}, has_audio=True))

    assert client.calls == 2
    assert message.id == 10


def test_audio_uses_imageio_ffmpeg_when_available(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: "/bundled/ffmpeg"),
    )

    audio = Audio(raw=b"data", name="voice.wav")

    assert audio.ffmpeg_path == "/bundled/ffmpeg"
