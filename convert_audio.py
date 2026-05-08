from pathlib import Path
import shutil
import subprocess


INPUT = Path("/home/jacquy-ngonga4/acia/mildiou1.opus")
MP3_OUTPUT = INPUT.with_suffix(".mp3")
M4A_OUTPUT = INPUT.with_suffix(".m4a")


def run_ffmpeg(args: list[str]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(
            "ffmpeg est introuvable. Installe-le avec : sudo apt-get install ffmpeg"
        )

    subprocess.run([ffmpeg, *args], check=True)


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Fichier introuvable : {INPUT}")

    run_ffmpeg(
        [
            "-y",
            "-i",
            str(INPUT),
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(MP3_OUTPUT),
        ]
    )

    run_ffmpeg(
        [
            "-y",
            "-i",
            str(INPUT),
            "-vn",
            "-codec:a",
            "aac",
            "-b:a",
            "128k",
            str(M4A_OUTPUT),
        ]
    )

    print(f"MP3 cree : {MP3_OUTPUT}")
    print(f"M4A cree : {M4A_OUTPUT}")


if __name__ == "__main__":
    main()
