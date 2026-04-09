import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def build_filter_chain(preset: str, denoise: bool) -> str:
    # These presets are tuned for ASR clarity, not pleasant listening.
    if preset == "strong":
        filters = [
            "highpass=f=130",
            "equalizer=f=170:t=h:w=180:g=-7",
            "equalizer=f=280:t=h:w=220:g=-3",
            "equalizer=f=3200:t=h:w=2600:g=4.5",
            "equalizer=f=6200:t=h:w=1800:g=2",
            "acompressor=threshold=-24dB:ratio=2.5:attack=5:release=60:makeup=4",
            "alimiter=limit=0.92",
        ]
        if denoise:
            filters.insert(5, "afftdn=nf=-28")
        return ",".join(filters)

    filters = [
        "highpass=f=100",
        "equalizer=f=180:t=h:w=170:g=-5",
        "equalizer=f=260:t=h:w=200:g=-2",
        "equalizer=f=3200:t=h:w=2400:g=3",
        "acompressor=threshold=-20dB:ratio=2.0:attack=5:release=50:makeup=2",
        "alimiter=limit=0.92",
    ]
    if denoise:
        filters.insert(4, "afftdn=nf=-24")
    return ",".join(filters)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_transcription.wav")


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    print("ffmpeg was not found in PATH.", file=sys.stderr)
    print("Install ffmpeg first, then run this script again.", file=sys.stderr)
    raise SystemExit(1)


def run_ffmpeg(input_path: Path, output_path: Path, preset: str, denoise: bool, overwrite: bool) -> None:
    filter_chain = build_filter_chain(preset, denoise)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vn",
        "-af",
        filter_chain,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Preset: {preset}")
    print(f"Denoise: {'on' if denoise else 'off'}")
    print()
    print("Running ffmpeg speech-preprocessing pipeline...")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print()
        print(f"ffmpeg failed with exit code {exc.returncode}.", file=sys.stderr)
        raise SystemExit(exc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess boomy audio into a speech-first WAV for transcription."
    )
    parser.add_argument("input", help="Path to the input audio/video file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the cleaned output WAV. Defaults to <input>_transcription.wav",
    )
    parser.add_argument(
        "--preset",
        choices=["mild", "strong"],
        default="mild",
        help="Use 'mild' first. Switch to 'strong' for very muddy or bass-heavy speech.",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable denoising if it makes speech sound watery.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_ffmpeg()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        raise SystemExit(1)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(input_path)
    )

    if output_path.exists() and not args.overwrite:
        print(f"Output already exists: {output_path}", file=sys.stderr)
        print("Use --overwrite or choose a different --output path.", file=sys.stderr)
        raise SystemExit(1)

    run_ffmpeg(
        input_path=input_path,
        output_path=output_path,
        preset=args.preset,
        denoise=not args.no_denoise,
        overwrite=args.overwrite,
    )

    print()
    print("Done.")
    print(f"Cleaned file ready for transcription: {output_path}")


if __name__ == "__main__":
    main()
