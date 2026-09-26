#!/bin/bash
# One v5 queue worker on a SageMaker notebook instance. Started in the background by
# the lifecycle OnStart hook; safe to rerun (reuses the venv and extracted data).
set -uo pipefail
BUCKET=${ER_BUCKET:-amazon-sagemaker-580857071542-ap-south-1-dn0trrxacr3ddt}
PREFIX=${ER_PREFIX:-shared/er-v5-20260927}
DATA_KEY=${ER_DATA_KEY:-shared/entity-v4-job-20260927/input/student_resource.zip}
REGION=ap-south-1
BASE=/home/ec2-user/SageMaker/er
mkdir -p "$BASE" && cd "$BASE"
exec >> "$BASE/bootstrap.log" 2>&1
NAME=$(python3 -c "import json;print(json.load(open('/opt/ml/metadata/resource-metadata.json'))['ResourceName'])" 2>/dev/null || hostname)
# Optional per-instance queue: s3://$BUCKET/$PREFIX/control/prefix-$NAME holds another prefix.
OVERRIDE=$(aws s3 cp "s3://$BUCKET/$PREFIX/control/prefix-$NAME" - --region $REGION 2>/dev/null | tr -d '[:space:]')
[ -n "$OVERRIDE" ] && PREFIX=$OVERRIDE
up() { aws s3 cp "$BASE/bootstrap.log" "s3://$BUCKET/$PREFIX/logs/$NAME.bootstrap.log" --sse AES256 --only-show-errors --region $REGION || true; }
trap up EXIT
echo "$(date -u) bootstrap $NAME nproc=$(nproc) mem=$(free -g | awk '/Mem:/{print $2}')G"
aws s3 cp "s3://$BUCKET/$PREFIX/code/code.tar.gz" code.tar.gz --only-show-errors --region $REGION || exit 1
rm -rf code && mkdir code && tar xzf code.tar.gz -C code || exit 1
if [ ! -x venv/bin/python ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$BASE/uvbin" INSTALLER_NO_MODIFY_PATH=1 sh
  UV="$BASE/uvbin/uv"; [ -x "$UV" ] || UV="$BASE/uvbin/bin/uv"
  if [ ! -x "$UV" ]; then python3 -m pip install --quiet --target "$BASE/uvpip" uv && UV="$BASE/uvpip/bin/uv"; fi
  "$UV" venv -p 3.12 venv || exit 1
  "$UV" pip install -p venv/bin/python -r code/requirements-v5.txt || exit 1
fi
echo "$(date -u) python ready: $(venv/bin/python --version)"
up
if [ ! -f data/student_resource/dataset/test/test_source3.tsv ]; then
  aws s3 cp "s3://$BUCKET/$DATA_KEY" data.zip --only-show-errors --region $REGION || exit 1
  mkdir -p data && (cd data && unzip -q -o ../data.zip 'student_resource/*' -x '*/.DS_Store') || exit 1
  rm -f data.zip
fi
echo "$(date -u) data ready: $(ls data/student_resource/dataset/*/ | tr '\n' ' ')"
up
export ER_WORKERS=$(nproc) POLARS_MAX_THREADS=$(nproc) OMP_NUM_THREADS=$(nproc) PYTHONUNBUFFERED=1 AWS_REGION=$REGION
failures=0
while [ $failures -lt 4 ]; do
  cd "$BASE/code"
  "$BASE/venv/bin/python" -m src.v5_dist worker --data "$BASE/data/student_resource/dataset" \
    --bucket "$BUCKET" --prefix "$PREFIX" --work "$BASE/work" --name "$NAME" --stay --reload \
    --logfile "$BASE/worker.log" >> "$BASE/worker.log" 2>&1
  status=$?
  cd "$BASE"
  echo "$(date -u) worker exited $status (failures $failures)"
  if [ $status -eq 75 ]; then
    # new code published: refresh and restart the worker without rebooting the instance
    aws s3 cp "s3://$BUCKET/$PREFIX/code/code.tar.gz" code.tar.gz --only-show-errors --region $REGION \
      && rm -rf code && mkdir code && tar xzf code.tar.gz -C code && echo "$(date -u) code reloaded"
    up
    continue
  fi
  [ $status -eq 0 ] && break
  failures=$((failures + 1))
  sleep 15
done
aws s3 cp "$BASE/worker.log" "s3://$BUCKET/$PREFIX/logs/$NAME.log" --sse AES256 --only-show-errors --region $REGION || true
if aws s3 ls "s3://$BUCKET/$PREFIX/control/stop-instances" --region $REGION >/dev/null 2>&1; then
  up
  aws sagemaker stop-notebook-instance --notebook-instance-name "$NAME" --region $REGION || true
fi
