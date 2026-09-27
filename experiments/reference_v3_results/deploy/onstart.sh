#!/bin/bash
# SageMaker notebook-instance lifecycle OnStart hook: fetch and background the worker bootstrap.
set -e
BUCKET=amazon-sagemaker-580857071542-ap-south-1-dn0trrxacr3ddt
PREFIX=shared/er-v5-20260927
mkdir -p /home/ec2-user/SageMaker
nohup bash -c "aws s3 cp s3://$BUCKET/$PREFIX/code/bootstrap.sh /home/ec2-user/SageMaker/bootstrap.sh --region ap-south-1 --only-show-errors && bash /home/ec2-user/SageMaker/bootstrap.sh" > /home/ec2-user/SageMaker/onstart.log 2>&1 &
exit 0
