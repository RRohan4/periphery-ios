# Building on the rented Mac

The whole loop: pull, build, drop an `.ipa` on the Desktop, upload it to Drive,
install it on the phone. Copy-paste from top to bottom.

Everything here runs **on the Mac**, not on the Linux box. The Linux box has no
Xcode, so it can compile nothing in this repo — it only edits and pushes.

---

## 0. Once per rented Mac

A rented Mac is a fresh machine every time, so this block is the setup tax. Skip
it if you have already done it on this instance.

```bash
xcode-select --install 2>/dev/null   # command line tools, if missing
sudo xcodebuild -license accept
git clone https://github.com/RRohan4/periphery-ios.git ~/periphery-ios
```

Then open the project **once** in Xcode so signing initialises:

```bash
open ~/periphery-ios/Periphery/Periphery.xcodeproj
```

Xcode → Signing & Capabilities → make sure the team is `6AU6S9Z86T` and
"Automatically manage signing" is ticked. You only need this for a *signed*
build; the unsigned `.ipa` path below does not care.

---

## 1. Pull

```bash
cd ~/periphery-ios
git pull origin main
```

If the pull touched `project.pbxproj` and Xcode is open, close and reopen it.
Xcode caches the project file and will otherwise build the old target list.

---

## 2. Build

The project uses a **file-system synchronized group**, so new `.swift` files are
picked up automatically. Nothing needs adding to the target.

### 2a. Just check it compiles

Fastest signal, no signing, no device:

```bash
cd ~/periphery-ios/Periphery
xcodebuild -project Periphery.xcodeproj -scheme Periphery \
           -destination 'generic/platform=iOS' \
           -configuration Release \
           CODE_SIGNING_ALLOWED=NO \
           build 2>&1 | grep -E "error:|warning:|BUILD"
```

Expect `** BUILD SUCCEEDED **`. Errors print with a file and line.

### 2b. Build the thing you actually install

```bash
cd ~/periphery-ios/Periphery
rm -rf ~/build
xcodebuild -project Periphery.xcodeproj -scheme Periphery \
           -destination 'generic/platform=iOS' \
           -configuration Release \
           -derivedDataPath ~/build \
           CODE_SIGNING_ALLOWED=NO \
           build
```

The app lands at:

```
~/build/Build/Products/Release-iphoneos/Periphery.app
```

---

## 3. Wrap it into an `.ipa` on the Desktop

An `.ipa` is a zip with the app inside a folder called `Payload`. That is the
entire format — there is nothing else to it.

```bash
cd ~/build/Build/Products/Release-iphoneos
rm -rf ~/Desktop/Payload ~/Desktop/Periphery.ipa
mkdir -p ~/Desktop/Payload
cp -R Periphery.app ~/Desktop/Payload/
cd ~/Desktop
zip -qry Periphery.ipa Payload
rm -rf Payload
ls -lh ~/Desktop/Periphery.ipa
```

`Periphery.ipa` is now sitting on the Desktop, ready to drag into Google Drive
in the browser.

### Or, all four steps in one

```bash
cd ~/periphery-ios && git pull origin main \
&& cd Periphery \
&& rm -rf ~/build \
&& xcodebuild -project Periphery.xcodeproj -scheme Periphery \
              -destination 'generic/platform=iOS' -configuration Release \
              -derivedDataPath ~/build CODE_SIGNING_ALLOWED=NO build \
&& rm -rf ~/Desktop/Payload ~/Desktop/Periphery.ipa \
&& mkdir -p ~/Desktop/Payload \
&& cp -R ~/build/Build/Products/Release-iphoneos/Periphery.app ~/Desktop/Payload/ \
&& (cd ~/Desktop && zip -qry Periphery.ipa Payload && rm -rf Payload) \
&& echo "READY: ~/Desktop/Periphery.ipa" && ls -lh ~/Desktop/Periphery.ipa
```

---

## 4. Onto the phone

The `.ipa` above is **unsigned**, which is what Sideloadly and AltStore want —
they re-sign it with your own Apple ID on the way in.

1. Drag `~/Desktop/Periphery.ipa` into Google Drive in the browser.
2. On the machine with your phone plugged in, download it.
3. Sideloadly (or AltStore) → pick the `.ipa` → your Apple ID → Start.
4. On the phone: Settings → General → VPN & Device Management → trust the
   developer certificate.

A free Apple ID signature **expires after 7 days**. Re-sideloading the same
`.ipa` renews it and keeps the app's data.

### The direct route, if the phone is plugged into the Mac

Skip all of section 3 and 4:

```bash
open ~/periphery-ios/Periphery/Periphery.xcodeproj
```

Select the phone as the destination, press ⌘R. Xcode signs and installs it.
This is faster, and the only way to get a debugger and console logs.

---

## 5. Check it actually works, in this order

Do these before driving anywhere. Each one fails loudly and cheaply.

| # | Tab | What to look for |
|---|---|---|
| 1 | **Self-check** | **8 rows, all green.** Not 6 — `focus of expansion` and the rest are new. If `focus of expansion` is red, the camera pitch estimator is producing wrong angles and nothing downstream is worth reading. |
| 2 | **Compute** | Every operation planned for the Neural Engine, both models. |
| 3 | **Latency** | Burst, 200 frames. Expect ~3.9 ms median on an A18. |
| 4 | **Flow** | Walk forward holding the phone. See below. |
| 5 | **Live** | Landscape, camera left, bird's-eye right. Horizon line should sit on the real horizon. |

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

**Build succeeds but the new tab is missing**
Xcode is holding a stale `project.pbxproj`. Quit Xcode, then:
```bash
rm -rf ~/build ~/Library/Developer/Xcode/DerivedData/Periphery-*
```
and build again.

**`No such file or directory: Periphery.app`**
The build failed further up but the shell kept going. Re-run section 2a on its
own and read the errors.

**App installs, then crashes at launch**
Almost always the models. Confirm both `.mlpackage`s made it into the bundle:
```bash
ls ~/build/Build/Products/Release-iphoneos/Periphery.app/*.mlmodelc
```
Expect `backbone_static.mlmodelc` and `head_static.mlmodelc`.

**Everything builds but the app is portrait**
It should not be — the app is landscape-locked now. If it rotates, the pull did
not include `project.pbxproj`. Check `git log --oneline -3`.
