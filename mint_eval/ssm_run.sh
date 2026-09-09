#!/usr/bin/env bash
# Run a shell script (read from stdin) on an EC2 instance via SSM Session Manager.
#
# Why SSM rather than ssh: a GPU box provisioned by an ops team often has no public IP and a key you
# do not hold. If the instance has the SSM agent and an instance profile allowing it, this needs
# neither. Everything below is driven by environment variables so no infrastructure identifier is
# baked into the file.
#
#   SSM_INSTANCE=i-xxxxxxxx AWS_PROFILE=myprofile AWS_DEFAULT_REGION=eu-west-1 \
#       ./ssm_run.sh <<'CMD'
#   nvidia-smi
#   CMD
set -euo pipefail
: "${SSM_INSTANCE:?set SSM_INSTANCE to the target instance id}"
T=$(mktemp /tmp/ssmparam.XXXXXX.json)
trap 'rm -f "$T"' EXIT
# build the parameter JSON with python: naive shell quoting mangles newlines into literal "n"
python3 -c '
import json,sys
sys.stdout.write(json.dumps({"commands": sys.stdin.read().split("\n")}))
' > "$T"
CID=$(aws ssm send-command --instance-ids "$SSM_INSTANCE" --document-name AWS-RunShellScript \
      --timeout-seconds "${SSM_TIMEOUT:-3600}" --parameters "file://$T" \
      --query 'Command.CommandId' --output text)
S=Pending
for _ in $(seq 1 "${SSM_WAIT:-240}"); do
  S=$(aws ssm get-command-invocation --command-id "$CID" --instance-id "$SSM_INSTANCE" \
      --query Status --output text 2>/dev/null || echo Pending)
  case "$S" in Success|Failed|TimedOut|Cancelled) break;; esac
  sleep 5
done
echo "[$S]"
aws ssm get-command-invocation --command-id "$CID" --instance-id "$SSM_INSTANCE" \
  --query StandardOutputContent --output text
E=$(aws ssm get-command-invocation --command-id "$CID" --instance-id "$SSM_INSTANCE" \
    --query StandardErrorContent --output text)
if [ -n "$E" ] && [ "$E" != "None" ]; then echo "--- stderr ---"; echo "$E"; fi
