#!/bin/bash
#
# Pull, build, and leave an .ipa somewhere you can drag into Google Drive.
#
# Run it from anywhere inside the repo:
#
#     ./tools/build-ipa.sh
#
# Nothing here assumes where the repo lives or what $HOME is. A rented Mac
# often has neither where you expect -- `~` expanded to /Users/u on one of
# them, which is how this script came to exist.
#
# Options:
#   --no-pull    build what is on disk, do not touch git
#   --check      compile only, no .ipa (fastest way to see errors)

set -euo pipefail

PULL=1
CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --no-pull) PULL=0 ;;
        --check)   CHECK_ONLY=1 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's|^# \{0,1\}||'; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

# --- where things are -------------------------------------------------------

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PROJECT="$REPO/Periphery/Periphery.xcodeproj"
[ -d "$PROJECT" ] || { echo "no project at $PROJECT" >&2; exit 1; }

# Build products go in the repo, not in $HOME: it is the one directory we have
# just proven exists.
BUILD="$REPO/.build"

# The Desktop is the point of the exercise, but only if there is one.
if [ -d "$HOME/Desktop" ]; then
    DEST="$HOME/Desktop"
elif [ -d "/Users/$(whoami)/Desktop" ]; then
    DEST="/Users/$(whoami)/Desktop"
else
    DEST="$REPO"
fi

echo "repo    $REPO"
echo "output  $DEST"
echo

# --- pull -------------------------------------------------------------------

if [ "$PULL" -eq 1 ]; then
    echo "==> git pull"
    git -C "$REPO" pull origin main
    echo
fi

# --- build ------------------------------------------------------------------

echo "==> xcodebuild"
rm -rf "$BUILD"
xcodebuild -project "$PROJECT" \
           -scheme Periphery \
           -destination 'generic/platform=iOS' \
           -configuration Release \
           -derivedDataPath "$BUILD" \
           CODE_SIGNING_ALLOWED=NO \
           build

APP="$BUILD/Build/Products/Release-iphoneos/Periphery.app"
[ -d "$APP" ] || { echo "build reported success but $APP is missing" >&2; exit 1; }

# The models are the usual reason a build succeeds and the app dies at launch.
for model in backbone_static head_static; do
    [ -e "$APP/$model.mlmodelc" ] \
        || echo "WARNING: $model.mlmodelc is not in the bundle" >&2
done

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo
    echo "compiled clean. no .ipa written (--check)."
    exit 0
fi

# --- wrap -------------------------------------------------------------------
#
# An .ipa is a zip with the app inside a folder called Payload. That is the
# entire format.

echo
echo "==> packaging"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/Payload"
cp -R "$APP" "$STAGE/Payload/"
rm -f "$DEST/Periphery.ipa"
( cd "$STAGE" && zip -qry "$DEST/Periphery.ipa" Payload )

echo
echo "READY: $DEST/Periphery.ipa"
ls -lh "$DEST/Periphery.ipa"
echo
echo "Unsigned, which is what Sideloadly and AltStore want -- they re-sign it"
echo "with your Apple ID on the way in."
