#!/usr/bin/env bash
# Start the Dash/Flask app in the background via flask CLI
export FLASK_APP=main:server
export FLASK_ENV=production
nohup flask run --host=0.0.0.0 --port=8050 > vox.log 2>&1 &
echo $! > vox.pid
echo "Vox Flask app started (PID=$(cat vox.pid)), logs in vox.log"
