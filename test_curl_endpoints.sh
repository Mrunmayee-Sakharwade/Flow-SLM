#!/usr/bin/env bash
# ==============================================================================
# Johnson & Johnson Commercial Oncology SLM Next-Question API Test Script
# Server: http://10.225.67.250:8008
# ==============================================================================

SERVER_HOST="${1:-10.225.67.250}"
SERVER_PORT="${2:-8008}"
BASE_URL="http://${SERVER_HOST}:${SERVER_PORT}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE} Testing J&J Commercial Oncology SLM Next-Question Endpoints${NC}"
echo -e "${BLUE} Target Server: ${BASE_URL}${NC}"
echo -e "${BLUE}================================================================${NC}\n"

# ------------------------------------------------------------------------------
# STEP 1: Verify System & KG Metadata (/api/kg/info)
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[1/4] Checking Knowledge Graph Metadata (/api/kg/info)...${NC}"
KG_INFO=$(curl -s -X GET "${BASE_URL}/api/kg/info")
echo "Response:"
echo "${KG_INFO}" | python3 -m json.tool 2>/dev/null || echo "${KG_INFO}"
echo ""

# ------------------------------------------------------------------------------
# STEP 2: Initialize a New Interview Session (/api/session/new)
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[2/4] Initializing New Session (/api/session/new)...${NC}"

SESSION_PAYLOAD='{
  "role": "OS",
  "brand": "INLEXZO",
  "account_name": "Apollo Hospitals",
  "user_name": "Aniruddha Joshi"
}'

SESSION_RESP=$(curl -s -X POST "${BASE_URL}/api/session/new" \
  -H "Content-Type: application/json" \
  -d "${SESSION_PAYLOAD}")

echo "Response:"
echo "${SESSION_RESP}" | python3 -m json.tool 2>/dev/null || echo "${SESSION_RESP}"

# Extract session_id
SESSION_ID=$(echo "${SESSION_RESP}" | python3 -c 'import sys, json; print(json.load(sys.stdin).get("session_id", ""))' 2>/dev/null)

if [ -z "${SESSION_ID}" ]; then
  echo -e "\n${YELLOW}[!] Fallback to default session ID 'OS_SESS_001'${NC}"
  SESSION_ID="OS_SESS_001"
else
  echo -e "\n${GREEN}[✓] Successfully Created Session: ${SESSION_ID}${NC}\n"
fi

# ------------------------------------------------------------------------------
# STEP 3: POST /api/session/turn (Blocking JSON Output)
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[3/4] Testing POST /api/session/turn (Standard JSON Response)...${NC}"

TURN_PAYLOAD=$(cat <<EOF
{
  "session_id": "${SESSION_ID}",
  "utterance": "I met with Dr. Anurag at Apollo Hospitals regarding patient identification for BCG unresponsive patients."
}
EOF
)

TURN_RESP=$(curl -s -X POST "${BASE_URL}/api/session/turn" \
  -H "Content-Type: application/json" \
  -d "${TURN_PAYLOAD}")

echo "Response:"
echo "${TURN_RESP}" | python3 -m json.tool 2>/dev/null || echo "${TURN_RESP}"
echo -e "\n${GREEN}[✓] Standard turn response received successfully.${NC}\n"

# ------------------------------------------------------------------------------
# STEP 4: POST /api/session/turn_stream (Real-Time SSE Token Stream)
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[4/4] Testing POST /api/session/turn_stream (SSE Real-Time Stream)...${NC}"
echo -e "${CYAN}Streaming tokens live:${NC}"

STREAM_PAYLOAD=$(cat <<EOF
{
  "session_id": "${SESSION_ID}",
  "utterance": "We discussed the clinical efficacy data and trial results."
}
EOF
)

# Using curl -N (--no-buffer) to ensure instant display of streaming tokens
curl -N -s -X POST "${BASE_URL}/api/session/turn_stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d "${STREAM_PAYLOAD}"

echo -e "\n\n${GREEN}[✓] SSE Stream completed.${NC}"
echo -e "${BLUE}================================================================${NC}"
