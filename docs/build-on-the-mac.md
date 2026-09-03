# Building on the rented Mac

The whole loop: pull, build, drop an `.ipa` on the Desktop, upload it to Drive,
install it on the phone.

**The short version, from anywhere inside the repo on the Mac:**

```bash
./tools/build-ipa.sh
```

Everything here runs **on the Mac**, not on the Linux box. The Linux box has no
Xcode, so it can compile nothing in this repo — it only edits and pushes.

---

## 0. Once per rented Mac

A rented Mac is a fresh machine every time, so this block is the setup tax. Skip
it if you have already done it on this instance.

```bash
xcode-select --install 2>/dev/null   # command line tools, if missing
sudo xcodebuild -license accept
git clone https://github.com/RRohan4/periphery-ios.git
cd periphery-ios
```

Then open the project **once** in Xcode so the scheme is generated:

```bash
open Periphery/Periphery.xcodeproj
```

> **Do not trust `~` on a rented Mac.** On one of these the prompt said
> `Renuka.Raina@...` while `$HOME` was `/Users/u`, so every `~/...` path in a
> pasted command pointed at a directory that did not exist. That is why the
> build is a script that works out its own paths rather than a command you
> paste. If you ever need to know where you actually are:
> `echo "$HOME"; pwd`.

Xcode → Signing & Capabilities → make sure the team is `6AU6S9Z86T` and
"Automatically manage signing" is ticked. You only need this for a *signed*
build; the unsigned `.ipa` path below does not care.

---

## 1. Pull, build, package — one command

From **anywhere inside the repo**:

```bash
./tools/build-ipa.sh
```

That is the whole loop. It pulls `main`, builds Release for device unsigned,
wraps the app into an `.ipa`, and leaves it on the Desktop. It prints where it
put it.

It works out the repo root from git and the Desktop from `$HOME`, falling back
to the repo if there is no Desktop, so it does not care where the repo was
cloned or what the rented Mac calls your home directory.

Two flags:

```bash
./tools/build-ipa.sh --check      # compile only, no .ipa — fastest error signal
./tools/build-ipa.sh --no-pull    # build what is on disk, do not touch git
```

`--check` is the one to reach for when you expect compile errors: it skips
packaging and stops at the first failure.

The project uses a **file-system synchronized group**, so new `.swift` files are
picked up automatically. Nothing needs adding to the target.

### What it does, if you would rather run it by hand

An `.ipa` is a zip with the app inside a folder called `Payload`. That is the
entire format — there is nothing else to it.

```bash
REPO="$(git rev-parse --show-toplevel)"
git -C "$REPO" pull origin main
rm -rf "$REPO/.build"
xcodebuild -project "$REPO/Periphery/Periphery.xcodeproj" -scheme Periphery \
           -destination 'generic/platform=iOS' -configuration Release \
           -derivedDataPath "$REPO/.build" CODE_SIGNING_ALLOWED=NO build
mkdir -p /tmp/Payload
cp -R "$REPO/.build/Build/Products/Release-iphoneos/Periphery.app" /tmp/Payload/
( cd /tmp && zip -qry "$HOME/Desktop/Periphery.ipa" Payload && rm -rf Payload )
ls -lh "$HOME/Desktop/Periphery.ipa"
```

---

## 2. Onto the phone

The `.ipa` above is **unsigned**, which is what Sideloadly and AltStore want —
they re-sign it with your own Apple ID on the way in.

1. Drag `Periphery.ipa` off the Desktop into Google Drive in the browser.
2. On the machine with your phone plugged in, download it.
3. Sideloadly (or AltStore) → pick the `.ipa` → your Apple ID → Start.
4. On the phone: Settings → General → VPN & Device Management → trust the
   developer certificate.

A free Apple ID signature **expires after 7 days**. Re-sideloading the same
`.ipa` renews it and keeps the app's data.

### The direct route, if the phone is plugged into the Mac

Skip section 1 and 2 entirely:

```bash
open "$(git rev-parse --show-toplevel)/Periphery/Periphery.xcodeproj"
```

Select the phone as the destination, press ⌘R. Xcode signs and installs it.
This is faster, and the only way to get a debugger and console logs.

---

## 3. Check it actually works, in this order

Do these before driving anywhere. Each one fails loudly and cheaply.

| # | Tab | What to look for |
|---|---|---|
| 1 | **Self-check** | **8 rows, all green.** Not 6 — `focus of expansion` and the rest are new. If `focus of expansion` is red, the camera pitch estimator is producing wrong angles and nothing downstream is worth reading. |
| 2 | **Compute** | Every operation planned for the Neural Engine, both models. |
| 3 | **Latency** | **Burst only.** Expect ~3.9 ms median on an A18. Do NOT run Sustained — see below. |
| 4 | **Flow** | Walk forward holding the phone. See below. |
| 5 | **Live** | Landscape, camera left, bird's-eye right. Horizon line should sit on the real horizon. |

### Do not run Latency → Sustained

The target sets `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, which makes
`enum Benchmark` implicitly main-actor isolated. The `Task.detached` that wraps
it therefore hops straight back to the main actor, defeating the comment above
it — so a ten-minute run freezes the UI and the watchdog may kill the app.
Swift 5 language mode reports this as a warning, not an error, which is why it
has always built and always been wrong.

Burst is unaffected. The fix is `nonisolated` on Benchmark, Detector and
Preprocessor, and it is unwritten because that chain cannot be verified without
a compiler.

### The Flow tab, on foot

This is the one that validates the camera pitch estimator without a car. It runs
in **handheld** mode, which cannot write the mount pose — the angle it measures
is the angle of your hand.

* **Walk forward.** The cyan cross should sit on the spot you are walking
  toward. Vectors green.
* **Tilt the phone down**, still walking the same way. The cross stays stuck to
  that spot in the world, riding up the frame as the whole scene does. Pitch
  goes **negative** by roughly however far you tilted. This single test proves
  the sign convention, the intrinsics and the roll unwind together.
* **Turn while walking.** If de-rotation works, the number barely moves. If it
  swings wildly, the gyro axis mapping is wrong.
* **Point at a passing person or car.** They light up **red** — outliers — with
  nothing in the app having recognised them as anything. That is the RANSAC
  consensus doing the job.

Switch the picker to **Driving** and it stops accumulating on foot: that profile
is gated at 8 m/s, which is the point.

---

## Troubleshooting

**`xcodebuild: error: Unable to find a destination`**
The scheme has not been generated yet. Open the project in Xcode once, then
retry.

**`error: Signing for "Periphery" requires a development team`**
You left out `CODE_SIGNING_ALLOWED=NO`. Add it back — the sideload path signs
later, not here.

**`cd: no such file or directory: /Users/u/...`**
`~` is not your home on this machine. Do not paste paths with `~` in them; run
`./tools/build-ipa.sh` from inside the repo, which derives its own.

**Build succeeds but the new tab is missing**
Xcode is holding a stale `project.pbxproj`. Quit Xcode, then:
```bash
rm -rf "$(git rev-parse --show-toplevel)/.build" \
       "$HOME/Library/Developer/Xcode/DerivedData/Periphery-"*
```
and build again.

**App installs, then crashes at launch**
Almost always the models. The script warns if they are missing; to check by
hand:
```bash
ls "$(git rev-parse --show-toplevel)/.build/Build/Products/Release-iphoneos/Periphery.app/"*.mlmodelc
```
Expect `backbone_static.mlmodelc` and `head_static.mlmodelc`.

**Everything builds but the app is portrait**
It should not be — the app is landscape-locked now. If it rotates, the pull did
not include `project.pbxproj`. Check `git log --oneline -3`.
