#!/usr/bin/env bash
# Restores the files that are not in git (source videos, frames, tiles, CVAT task zips, intermediate-stage
# videos, files over 50 MB; full list in data_manifest.txt) from the project archive on Google Drive. The
# archive is the whole project folder; files that already exist (everything from git) are never overwritten.
#
# Usage: bash download_data.sh [options]
#   --id ID          Google Drive file id of the archive (default: $DATA_DRIVE_ID or DEFAULT_ID below)
#   --archive PATH   use an archive you already downloaded (e.g. in a browser) instead of downloading it
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

is_tar() {  # a tar archive has the "ustar" magic at byte 257 (an HTML error page does not)
  [[ "$(dd if="$1" bs=1 skip=257 count=5 2>/dev/null)" == "ustar" ]]
}

if [[ $CHECK -eq 1 ]]; then
  echo "missing: $(count_missing) of $TOTAL files from data_manifest.txt"
  exit 0
fi

size_of() { [[ -f "$1" ]] && wc -c < "$1" | tr -d ' ' || echo 0; }

if [[ -z "$ARCHIVE" ]]; then
  # Direct link behind Drive's "file is too large to scan for viruses, download anyway" button.
  # Google serves the bytes only to range requests (a plain GET returns the warning page again),
  # so every request asks for "bytes=<already downloaded>-" and appends to the file.
  URL="${DATA_URL:-https://drive.usercontent.google.com/download?id=${ID}&export=download&confirm=t}"
  ARCHIVE="$DEST/aerial-vehicle-detector-pipeline.tar"
  HDR="$(mktemp)"
  trap 'rm -f "$HDR"' EXIT
  # total size from a 1-byte range request ("Content-Range: bytes 0-0/<total>")
  REMOTE=$(curl -sSL -r 0-0 -D - -o /dev/null "$URL" 2>/dev/null | tr -d '\r' \
           | awk -F/ 'tolower($0) ~ /^content-range:/ {s = $2} END {print s}') || REMOTE=""
  if [[ -z "$REMOTE" ]]; then
    echo "Google did not report the archive size (daily download limit?)." >&2
    echo "Try again later, or download it in a browser from https://drive.google.com/open?id=$ID" >&2
    echo "and run: bash download_data.sh --archive <path to the .tar>" >&2
    exit 1
  fi
  # a leftover that is not the start of a tar (e.g. a saved HTML page) cannot be resumed
  if [[ -f "$ARCHIVE" ]] && ! is_tar "$ARCHIVE"; then rm -f "$ARCHIVE"; fi
  echo "downloading the archive ($REMOTE bytes) to $ARCHIVE; run again to resume if it stops"
  for attempt in $(seq 1 20); do
    LOCAL=$(size_of "$ARCHIVE")
    [[ "$LOCAL" -lt "$REMOTE" ]] || break
    [[ "$LOCAL" -eq 0 ]] || echo "resuming from byte $LOCAL (attempt $attempt)"
    curl -L --progress-bar -r "${LOCAL}-" -D "$HDR" "$URL" >> "$ARCHIVE" || true
    STATUS=$(tr -d '\r' < "$HDR" | awk 'toupper($1) ~ /^HTTP/ {c = $2} END {print c}')
    if [[ "$STATUS" != "206" ]]; then
      # not a partial-content answer: cut off whatever was appended and wait a little
      dd if=/dev/null of="$ARCHIVE" bs=1 seek="$LOCAL" 2>/dev/null
      echo "server answered ${STATUS:-nothing} instead of 206, retrying in 10 s" >&2
      sleep 10
    fi
  done
  LOCAL=$(size_of "$ARCHIVE")
  if [[ "$LOCAL" -ne "$REMOTE" ]]; then
    echo "downloaded $LOCAL of $REMOTE bytes: run the script again to resume" >&2
    exit 1
  fi
  DOWNLOADED=1
fi

if ! is_tar "$ARCHIVE"; then
  echo "$ARCHIVE is not a tar archive: Google may have returned a page instead (daily download limit)." >&2
  echo "Try again later, or download it in a browser from https://drive.google.com/open?id=$ID" >&2
  echo "and run: bash download_data.sh --archive <path to the .tar>" >&2
  [[ $DOWNLOADED -eq 1 ]] && rm -f "$ARCHIVE"
  exit 1
fi

# GNU tar keeps existing files with --skip-old-files, bsdtar (macOS) with -k.
if tar --version 2>/dev/null | grep -q "GNU tar"; then KEEP_OLD="--skip-old-files"; else KEEP_OLD="-k"; fi
echo "extracting into $DEST (files from git are kept as they are)..."
tar -xf "$ARCHIVE" -C "$DEST" --strip-components=1 "$KEEP_OLD" || {
  echo "extraction failed: the archive may be incomplete, run the script again to resume the download" >&2
  exit 1
}

MISSING=$(count_missing)
echo "restored: $((TOTAL - MISSING)) of $TOTAL files from data_manifest.txt"
if [[ $DOWNLOADED -eq 1 && $KEEP -eq 0 ]]; then
  rm -f "$ARCHIVE"
  echo "deleted the downloaded archive (use --keep-archive to keep it)"
fi
[[ $MISSING -eq 0 ]] || { echo "some files are still missing: check the archive" >&2; exit 1; }
