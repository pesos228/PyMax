import asyncio
import mimetypes
import struct
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from aiofiles import open as aio_open
from aiohttp import ClientSession
from typing_extensions import override


class BaseFile(ABC):
    def __init__(
        self, raw: bytes | None = None, *, url: str | None = None, path: str | None = None
    ) -> None:
        self.raw = raw
        self.url = url
        self.path = path

        sources_count = sum(
            source is not None for source in (self.raw, self.url, self.path)
        )
        if sources_count == 0:
            raise ValueError("Either raw, url, or path must be provided.")

        if sources_count > 1:
            raise ValueError("Only one of raw, url, or path must be provided.")

    @abstractmethod
    async def read(self) -> bytes:
        if self.raw is not None:
            return self.raw

        if self.url:
            async with (
                ClientSession() as session,
                session.get(self.url) as response,
            ):
                response.raise_for_status()
                return await response.read()
        elif self.path:
            async with aio_open(self.path, "rb") as f:
                return await f.read()
        else:
            raise ValueError("Either url or path must be provided.")


@dataclass(slots=True)
class PreparedAudio:
    path: Path
    file_name: str
    duration_ms: int
    wave: bytes


class Photo(BaseFile):
    ALLOWED_EXTENSIONS: ClassVar[set[str]] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
    }  # FIXME: костыль ✅

    def __init__(
        self,
        raw: bytes | None = None,
        *,
        url: str | None = None,
        path: str | None = None,
        name: str | None = None,
    ) -> None:
        if path:
            self.file_name = Path(path).name
        elif url:
            self.file_name = Path(url).name
        elif name:
            self.file_name = name
        else:
            self.file_name = ""

        super().__init__(raw=raw, url=url, path=path)

    def validate_photo(self) -> tuple[str, str] | None:
        if self.path:
            extension = Path(self.path).suffix.lower()
            if extension not in self.ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"Invalid photo extension: {extension}. Allowed: {self.ALLOWED_EXTENSIONS}"
                )

            return (extension[1:], ("image/" + extension[1:]).lower())
        elif self.url:
            extension = Path(self.url).suffix.lower()
            if extension not in self.ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"Invalid photo extension in URL: {extension}. Allowed: {self.ALLOWED_EXTENSIONS}"
                )

            mime_type = mimetypes.guess_type(self.url)[0]

            if not mime_type or not mime_type.startswith("image/"):
                raise ValueError(f"URL does not appear to be an image: {self.url}")

            return (extension[1:], mime_type)
        return None

    @override
    async def read(self) -> bytes:
        return await super().read()


class Video(BaseFile):
    def __init__(
        self, raw: bytes | None = None, *, url: str | None = None, path: str | None = None
    ) -> None:
        self.file_name: str = ""
        if path:
            self.file_name = Path(path).name
        elif url:
            self.file_name = Path(url).name

        if not self.file_name:
            raise ValueError("Either url or path must be provided.")
        super().__init__(raw=raw, url=url, path=path)

    @override
    async def read(self) -> bytes:
        return await super().read()


class Audio(BaseFile):
    CHUNK_SIZE: ClassVar[int] = 1024 * 1024
    PCM_SAMPLE_RATE: ClassVar[int] = 8000
    OPUS_SAMPLE_RATE: ClassVar[int] = 48000
    WAVE_SAMPLES: ClassVar[int] = 80
    WAVE_MAX_VALUE: ClassVar[int] = 100

    def __init__(
        self,
        raw: bytes | None = None,
        *,
        url: str | None = None,
        path: str | None = None,
        name: str | None = None,
        duration_ms: int | None = None,
        wave: bytes | None = None,
        convert: bool = True,
        ffmpeg_path: str | None = None,
    ) -> None:
        if path:
            self.file_name = Path(path).name
        elif url:
            self.file_name = Path(url).name
        elif name:
            self.file_name = name
        else:
            self.file_name = "voice.ogg"

        self.duration_ms = duration_ms
        self.wave = wave
        self.convert = convert
        self.ffmpeg_path = ffmpeg_path or self._resolve_ffmpeg_path()
        self._prepared: PreparedAudio | None = None
        self._temp_paths: list[Path] = []

        super().__init__(raw=raw, url=url, path=path)

    @override
    async def read(self) -> bytes:
        return await super().read()

    async def prepare(self) -> PreparedAudio:
        if self._prepared is not None:
            return self._prepared

        source = await self._materialize_source()
        if self.convert:
            audio_path = await self._convert_to_ogg_opus(source)
        else:
            self._ensure_ogg_opus(source)
            audio_path = source

        duration = self.duration_ms or self._parse_ogg_opus_duration_ms(audio_path)
        if duration is None:
            raise ValueError(
                "Unable to determine audio duration; pass duration_ms explicitly."
            )

        wave = (
            self._normalize_wave(self.wave)
            if self.wave
            else await self._build_wave(audio_path, duration)
        )
        self._prepared = PreparedAudio(
            path=audio_path,
            file_name=Path(audio_path).name,
            duration_ms=duration,
            wave=wave,
        )
        return self._prepared

    def cleanup(self) -> None:
        for path in self._temp_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        self._temp_paths.clear()
        self._prepared = None

    async def _materialize_source(self) -> Path:
        if self.path:
            return Path(self.path)

        suffix = Path(self.file_name).suffix or ".audio"
        target = self._new_temp_path(suffix=suffix)

        if self.raw is not None:
            async with aio_open(target, "wb") as file:
                await file.write(self.raw)
            return target

        if self.url:
            async with ClientSession() as session:
                async with session.get(self.url) as response:
                    response.raise_for_status()
                    async with aio_open(target, "wb") as file:
                        async for chunk in response.content.iter_chunked(
                            self.CHUNK_SIZE
                        ):
                            await file.write(chunk)
            return target

        raise ValueError("Either url or path must be provided.")

    async def _convert_to_ogg_opus(self, source: Path) -> Path:
        output = self._new_temp_path(suffix=".ogg")
        returncode, stderr = await self._run_process(
            self.ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.OPUS_SAMPLE_RATE),
            "-c:a",
            "libopus",
            "-b:a",
            "48k",
            "-application",
            "audio",
            str(output),
        )
        if returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(
                f"ffmpeg failed to convert audio to Ogg Opus: {message}"
            )
        self._ensure_ogg_opus(output)
        return output

    async def _build_wave(self, audio_path: Path, duration_ms: int) -> bytes:
        try:
            return await self._build_wave_with_ffmpeg(audio_path, duration_ms)
        except Exception:
            return self._placeholder_wave()

    async def _build_wave_with_ffmpeg(self, audio_path: Path, duration_ms: int) -> bytes:
        total_samples = max(1, duration_ms * self.PCM_SAMPLE_RATE // 1000)
        proc = await asyncio.create_subprocess_exec(
            self.ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(self.PCM_SAMPLE_RATE),
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if proc.stdout is None:
            raise ValueError("ffmpeg stdout is unavailable")

        peaks = [0] * self.WAVE_SAMPLES
        sample_index = 0
        pending = b""
        while True:
            chunk = await proc.stdout.read(self.CHUNK_SIZE)
            if not chunk:
                break
            data = pending + chunk
            even_len = len(data) - (len(data) % 2)
            pending = data[even_len:]

            for (sample,) in struct.iter_unpack("<h", data[:even_len]):
                idx = min(
                    self.WAVE_SAMPLES - 1,
                    sample_index * self.WAVE_SAMPLES // total_samples,
                )
                peaks[idx] = max(peaks[idx], abs(sample))
                sample_index += 1

        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"ffmpeg failed to build waveform: {message}")

        max_peak = max(peaks) or 1
        return bytes(
            max(
                1,
                min(
                    self.WAVE_MAX_VALUE,
                    round(peak * self.WAVE_MAX_VALUE / max_peak),
                ),
            )
            for peak in peaks
        )

    async def _run_process(self, *args: str) -> tuple[int, bytes]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                f"{self.ffmpeg_path!r} is required for audio conversion but was "
                "not found. Install imageio-ffmpeg or system ffmpeg."
            ) from exc
        _, stderr = await proc.communicate()
        return proc.returncode or 0, stderr

    @staticmethod
    def _resolve_ffmpeg_path() -> str:
        try:
            import imageio_ffmpeg
        except ImportError:
            return "ffmpeg"

        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

    def _new_temp_path(self, suffix: str) -> Path:
        handle = tempfile.NamedTemporaryFile(
            prefix="pymax-audio-",
            suffix=suffix,
            delete=False,
        )
        path = Path(handle.name)
        handle.close()
        self._temp_paths.append(path)
        return path

    def _ensure_ogg_opus(self, path: Path) -> None:
        with path.open("rb") as file:
            head = file.read(4096)
        if not head.startswith(b"OggS"):
            raise ValueError("Audio must be an Ogg container: missing OggS header.")
        if b"OpusHead" not in head:
            raise ValueError("Audio must be Ogg Opus: missing OpusHead header.")

    def _parse_ogg_opus_duration_ms(self, path: Path) -> int | None:
        pre_skip = 0
        last_granule = None

        with path.open("rb") as file:
            while True:
                header = file.read(27)
                if not header:
                    break
                if len(header) < 27 or header[:4] != b"OggS":
                    break

                granule = struct.unpack_from("<Q", header, 6)[0]
                page_segments = header[26]
                segment_sizes = file.read(page_segments)
                if len(segment_sizes) < page_segments:
                    break

                body_size = sum(segment_sizes)
                body = file.read(body_size)
                if len(body) < body_size:
                    break

                opus_head_pos = body.find(b"OpusHead")
                if opus_head_pos != -1 and opus_head_pos + 12 <= len(body):
                    pre_skip = struct.unpack_from("<H", body, opus_head_pos + 10)[0]

                if granule != 0xFFFFFFFFFFFFFFFF:
                    last_granule = granule

        if last_granule is None:
            return None

        samples = max(0, int(last_granule) - pre_skip)
        return max(1, round(samples * 1000 / self.OPUS_SAMPLE_RATE))

    def _normalize_wave(self, wave: bytes) -> bytes:
        if len(wave) != self.WAVE_SAMPLES:
            raise ValueError(
                f"Audio wave must be exactly {self.WAVE_SAMPLES} bytes."
            )
        return bytes(min(self.WAVE_MAX_VALUE, value) for value in wave)

    def _placeholder_wave(self) -> bytes:
        values = []
        for i in range(self.WAVE_SAMPLES):
            phase = i % 20
            peak = phase if phase <= 10 else 20 - phase
            values.append(20 + peak * 6)
        return bytes(values)


Voice = Audio


class File(BaseFile):
    def __init__(
        self, raw: bytes | None = None, *, url: str | None = None, path: str | None = None
    ) -> None:
        self.file_name: str = ""
        if path:
            self.file_name = Path(path).name
        elif url:
            self.file_name = Path(url).name

        if not self.file_name:
            raise ValueError("Either url or path must be provided.")

        super().__init__(raw=raw, url=url, path=path)

    @override
    async def read(self) -> bytes:
        return await super().read()
