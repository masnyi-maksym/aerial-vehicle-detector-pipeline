#!/usr/bin/env bash
# Unpacks the files that are not in git (source videos, frames, tiles, CVAT task zips, intermediate-stage
# videos, files over 50 MB; full list in data_manifest.txt) from the project archive into their places.
# Download the archive first (a browser is needed: Google asks to confirm the download of a large file):
#   https://drive.google.com/open?id=163_on3Jn_WM82-ih17FeXchNMKZXFMMM
# and put aerial-vehicle-detector-pipeline.tar into the repository root. Files that already exist
# (everything from git) are never overwritten.
#
# Usage: bash restore_data.sh [options]
#   --archive PATH    the archive (default: aerial-vehicle-detector-pipeline.tar in the repository root)
#   --dest DIR        project root to restore into (default: the folder of this script)
#   --expect-bytes N  size of the complete archive, 0 = do not check (default: EXPECT_BYTES below)
#   --delete-archive  delete the archive after a successful restore (default: keep it)
#   --check-only      only report how many files from data_manifest.txt are missing
set -euo pipefail

ARCHIVE_URL="https://drive.google.com/open?id=163_on3Jn_WM82-ih17FeXchNMKZXFMMM"
EXPECT_BYTES=24637757440
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE=""
DELETE=0
CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) ARCHIVE="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --expect-bytes) EXPECT_BYTES="$2"; shift 2 ;;
    --delete-archive) DELETE=1; shift ;;
    --check-only) CHECK=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
ARCHIVE="${ARCHIVE:-$DEST/aerial-vehicle-detector-pipeline.tar}"

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

if [[ ! -f "$ARCHIVE" ]]; then
  echo "no archive at $ARCHIVE" >&2
  echo "download it in a browser from $ARCHIVE_URL and put it into $DEST (or pass --archive <path>)" >&2
  exit 1
fi
SIZE=$(wc -c < "$ARCHIVE" | tr -d ' ')
if [[ "$EXPECT_BYTES" -gt 0 && "$SIZE" -ne "$EXPECT_BYTES" ]]; then
  echo "$ARCHIVE has $SIZE bytes, the complete archive has $EXPECT_BYTES: the download is not finished" >&2
  echo "or broken. Download it again from $ARCHIVE_URL" >&2
  exit 1
fi
# a tar archive has the "ustar" magic at byte 257 (a saved HTML page does not)
if [[ "$(dd if="$ARCHIVE" bs=1 skip=257 count=5 2>/dev/null)" != "ustar" ]]; then
  echo "$ARCHIVE is not a tar archive. Download it again from $ARCHIVE_URL" >&2
  exit 1
fi

# GNU tar keeps existing files with --skip-old-files, bsdtar (macOS) with -k.
if tar --version 2>/dev/null | grep -q "GNU tar"; then KEEP_OLD="--skip-old-files"; else KEEP_OLD="-k"; fi
echo "unpacking $ARCHIVE into $DEST (files from git are kept as they are), a few minutes..."
tar -xf "$ARCHIVE" -C "$DEST" --strip-components=1 "$KEEP_OLD"

MISSING=$(count_missing)
echo "restored: $((TOTAL - MISSING)) of $TOTAL files from data_manifest.txt"
[[ $MISSING -eq 0 ]] || { echo "some files are still missing: is this the right archive?" >&2; exit 1; }
if [[ $DELETE -eq 1 ]]; then
  rm -f "$ARCHIVE"
  echo "deleted $ARCHIVE"
fi
