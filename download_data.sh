#!/usr/bin/env bash
# Pull the permitted challenge data (Overture extracts, ACS housing, strata, boundaries).
# The removed reference layers are not in the bucket and are not used anywhere.
set -euo pipefail
DEST="${BB_DATA_DIR:-$(dirname "$0")/data}"
BUCKET=s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge
mkdir -p "$DEST"
aws s3 sync "$BUCKET/reference/" "$DEST/reference/" --no-sign-request \
  --exclude '*roads-unfiltered*' --exclude '*.csv' \
  --include '*sample-submission.csv' --include '*acs-housing.csv'
for r in maricopa-az northern-ca eastern-ok south-central-tx eastern-wa; do
  aws s3 sync "$BUCKET/strata/$r/" "$DEST/strata/$r/" --no-sign-request --exclude '*.csv'
done
aws s3 sync "$BUCKET/boundaries/" "$DEST/boundaries/" --no-sign-request
echo "data in $DEST"
