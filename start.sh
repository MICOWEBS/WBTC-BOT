#!/bin/bash

# WBTC Scalp Bot Startup Script

# Exit on error
set -e

echo "Starting WBTC Scalp Bot..."

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for environment file
if [ ! -f ".env" ]; then
    echo "Environment file not found. Creating from example..."
    cp .env.example .env
    echo "Please edit .env file with your credentials before running again."
    exit 1
fi

# Check parameters
START_BOT=false
START_API=true

# Process command line arguments
for arg in "$@"
do
    case $arg in
        --with-bot)
        START_BOT=true
        shift
        ;;
        --api-only)
        START_BOT=false
        START_API=true
        shift
        ;;
        *)
        # Unknown option
        ;;
    esac
done

# Start API server
if [ "$START_API" = true ]; then
    echo "Starting API server..."
    python api_server.py &
    API_PID=$!
    echo "API server started with PID: $API_PID"
fi

# Start trading bot if requested
if [ "$START_BOT" = true ]; then
    echo "Starting trading bot..."
    python advanced_trader.py &
    BOT_PID=$!
    echo "Trading bot started with PID: $BOT_PID"
fi

echo "Startup complete! Use Ctrl+C to stop all processes."

# Wait for interruption
wait 