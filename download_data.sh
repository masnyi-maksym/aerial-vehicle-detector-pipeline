#!/usr/bin/env bash
# Restores the files that are not in git (source videos, frames, tiles, CVAT task zips, intermediate-stage
# videos, files over 50 MB; full list in data_manifest.txt) from the project archive on Google Drive. The
# archive is the whole project folder; files that already exist (everything from git) are never overwritten.
#
# Usage: bash download_data.sh [options]
#   --id ID          Google Drive file id or link of the archive (default: $DATA_DRIVE_ID or DEFAULT_ID below)
#   --archive PATH   use an archive you already downloaded instead of downloading it
#   --dest DIR       project root to restore into (default: the folder of this script)
#   --keep-archive   keep the downloaded .tar (default: delete it after extracting)
#   --check-only     only report how many files from data_manifest.txt are missing
# Needs ~23 GB for the download plus ~22 GB for the extracted files. An interrupted download resumes
# when the script is run again.
set -euo pipefail

DEFAULT_ID="163_on3Jn_WM82-ih17FeXchNMKZXFMMM"   # https://drive.google.com/open?id=163_on3Jn_WM82-ih17FeXchNMKZXFMMM
ID="${DATA_DRIVE_ID:-$DEFAULT_ID}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE=""
KEEP=0
CHECK=0
DOWNLOADED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --id) ID="$2"; shift 2 ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --keep-archive) KEEP=1; shift ;;
    --check-only) CHECK=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

MANIFEST="$DEST/data_manifest.txt"
[[ -f "$MANIFEST" ]] || { echo "no $MANIFEST: run the script from the cloned repository" >&2; exit 2; }
TOTAL=$(grep -c . "$MANIFEST")

count_missing() {
  local n=0 f
  while IFS= read -r f; do [[ -z "$f" || -e "$DEST/$f" ]] || n=$((n + 1)); done < "$MANIFEST"
  echo "$n"
}

if [[ $CHECK -eq 1 ]]; then
  echo "missing: $(count_missing) of $TOTAL files from data_manifest.txt"
  exit 0
fi

if [[ -z "$ARCHIVE" ]]; then
  if [[ "$ID" == "PUT_DRIVE_FILE_ID_HERE" ]]; then
    echo "set the Google Drive id of the archive: --id <id or link>, or DATA_DRIVE_ID" >&2
    exit 2
  fi
  if ! command -v gdown >/dev/null 2>&1; then
    echo "installing gdown (Google Drive downloader)..."
    python3 -m pip install --user --quiet gdown
    export PATH="$HOME/.local/bin:$PATH"
  fi
  ARCHIVE="$DEST/aerial-vehicle-detector-pipeline.tar"
  echo "downloading the archive (~23 GB) to $ARCHIVE; run again to resume if it stops"
  gdown --continue --retries 5 "$ID" -O "$ARCHIVE"
  DOWNLOADED=1
fi

# GNU tar keeps existing files with --skip-old-files, bsdtar (macOS) with -k.
if tar --version 2>/dev/null | grep -q "GNU tar"; then KEEP_OLD="--skip-old-files"; else KEEP_OLD="-k"; fi
echo "extracting into $DEST (files from git are kept as they are)..."
tar -xf "$ARCHIVE" -C "$DEST" --strip-components=1 "$KEEP_OLD"

MISSING=$(count_missing)
echo "restored: $((TOTAL - MISSING)) of $TOTAL files from data_manifest.txt"
if [[ $DOWNLOADED -eq 1 && $KEEP -eq 0 ]]; then
  rm -f "$ARCHIVE"
  echo "deleted the downloaded archive (use --keep-archive to keep it)"
fi
[[ $MISSING -eq 0 ]] || { echo "some files are still missing: check the archive id" >&2; exit 1; }
