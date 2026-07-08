#!/bin/zsh
# One-shot installer for the TradingAgents desk:
#   1. launchd agent com.overnightdesk.tradingagents — server + worker, always on
#      (RunAtLoad + KeepAlive; /bin/zsh already has Full Disk Access for launchd)
#   2. ~/Applications/TradingAgents Desk.app — opens the pixel floor whenever
# Rerun any time; it replaces both cleanly.
set -euo pipefail

DESK="/Users/fletcherlee/Documents/PERIS/PERIS/overnight-desk"
UV="/Users/fletcherlee/anaconda3/bin/uv"
LABEL="com.overnightdesk.tradingagents"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PORT=8102

# ---------------------------------------------------------------- launchd agent
chmod +x "$DESK/jobs/run_tradingagents_desk.sh"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$DESK/jobs/run_tradingagents_desk.sh</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>15</integer>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart "gui/$(id -u)/$LABEL"
echo "launchd agent $LABEL installed and started"

# ---------------------------------------------------------------- .app bundle
APP="$HOME/Applications/TradingAgents Desk.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>TradingAgents Desk</string>
  <key>CFBundleDisplayName</key><string>TradingAgents Desk</string>
  <key>CFBundleIdentifier</key><string>com.overnightdesk.tradingagents.desk</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>desk</string>
  <key>CFBundleIconFile</key><string>icon</string>
</dict>
</plist>
EOF

cat > "$APP/Contents/MacOS/desk" <<EOF
#!/bin/zsh
# TradingAgents Desk launcher: make sure the background server is up, then open the floor.
PORT=$PORT
LABEL=$LABEL
if ! curl -s -m 2 "http://localhost:\$PORT/state" > /dev/null; then
  launchctl kickstart "gui/\$(id -u)/\$LABEL" 2>/dev/null || \
    launchctl bootstrap "gui/\$(id -u)" "$PLIST" 2>/dev/null || true
  for i in {1..30}; do
    curl -s -m 1 "http://localhost:\$PORT/state" > /dev/null && break
    sleep 1
  done
fi
open "http://localhost:\$PORT/"
EOF
chmod +x "$APP/Contents/MacOS/desk"

# pixel icon: stdlib PNG -> iconset -> icns
TMP_ICON="$(mktemp -d)"
"$UV" run --directory "$DESK" python jobs/make_desk_icon.py "$TMP_ICON/icon512.png"
mkdir -p "$TMP_ICON/icon.iconset"
for size in 16 32 64 128 256 512; do
  sips -z $size $size "$TMP_ICON/icon512.png" \
    --out "$TMP_ICON/icon.iconset/icon_${size}x${size}.png" > /dev/null
done
cp "$TMP_ICON/icon.iconset/icon_512x512.png" "$TMP_ICON/icon.iconset/icon_256x256@2x.png"
iconutil -c icns "$TMP_ICON/icon.iconset" -o "$APP/Contents/Resources/icon.icns"
rm -rf "$TMP_ICON"

echo "installed: $APP"
echo "open it from ~/Applications (or Spotlight: 'TradingAgents Desk')"
