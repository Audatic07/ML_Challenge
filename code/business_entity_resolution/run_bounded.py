"""Run one training job with a wall-time cap, private backups and self shutdown.

Use within a dedicated Studio JupyterLab app. All runtime paths are arguments;
the script never prints credentials. A second process enforces the deadline even
if the training wrapper fails. EBS remains intact when DeleteApp stops compute.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def stop_app(args):
    import boto3
    client = boto3.client('sagemaker',region_name=args.region)
    client.delete_app(DomainId=args.domain,SpaceName=args.space,AppType='JupyterLab',AppName='default')


def backup(args, force=False):
    import boto3
    s3 = boto3.client('s3',region_name=args.region)
    names = ['run_manifest.json','model_manifest.json','model.txt','tune_retrieval_metrics.json',
             'tune_per_query.parquet','tune_scores.parquet','complete.json','job_status.json','train.log']
    index = args.work/'uploaded.json'
    uploaded = json.loads(index.read_text()) if index.exists() else {}
    changed = False
    for name in names:
        path = args.work / name
        if path.exists() and (force or uploaded.get(name) != [path.stat().st_size,path.stat().st_mtime_ns]):
            s3.upload_file(str(path),args.bucket,f'{args.prefix}/{name}',ExtraArgs={'ServerSideEncryption':'AES256'})
            uploaded[name]=[path.stat().st_size,path.stat().st_mtime_ns]
            changed=True
    if changed:
        index.write_text(json.dumps(uploaded,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',required=True)
    p.add_argument('--space',required=True)
    p.add_argument('--region',default='ap-south-1')
    p.add_argument('--bucket',required=True)
    p.add_argument('--prefix',required=True)
    p.add_argument('--work',required=True,type=Path)
    p.add_argument('--max-seconds',type=int,default=10800)
    p.add_argument('--watchdog',action='store_true')
    args, command = p.parse_known_args()
    if args.watchdog:
        time.sleep(args.max_seconds)
        # Stop dedicated compute even if the main wrapper was killed by OOM.
        for attempt in range(5):
            try:
                stop_app(args)
                return
            except Exception:
                time.sleep(10)
        raise RuntimeError('Watchdog failed to stop the dedicated Studio app')
    if command and command[0] == '--':
        command = command[1:]
    if not command or not 60 <= args.max_seconds <= 21600:
        p.error('Provide a training command and a 60..21600 second cap')
    args.work.mkdir(parents=True,exist_ok=True)
    lock = args.work/'job.lock'
    # Exclusive creation stops double-paste duplicate jobs. A failed lock requires
    # inspection of the recorded process, never an automatic retry/removal.
    with lock.open('x') as stream:
        json.dump({'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},stream)
    flags = ['--domain',args.domain,'--space',args.space,'--region',args.region,'--bucket',args.bucket,
             '--prefix',args.prefix,'--work',str(args.work),'--max-seconds',str(args.max_seconds)]
    watchdog = subprocess.Popen([sys.executable,__file__,*flags,'--watchdog'],start_new_session=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1',POLARS_MAX_THREADS='8',ER_WORKERS='8',OMP_NUM_THREADS='8')
    status = {'command':command,'max_seconds':args.max_seconds,'watchdog_pid':watchdog.pid,'state':'running'}
    (args.work/'job_status.json').write_text(json.dumps(status,indent=2))
    started = time.monotonic()
    with (args.work/'train.log').open('a',buffering=1) as log:
        child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            while child.poll() is None:
                if time.monotonic()-started > args.max_seconds-90:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                    status['state']='timed_out'
                    break
                try:
                    backup(args)
                except Exception as error:
                    log.write(f'Backup warning: {type(error).__name__}\n')
                time.sleep(30)
            status.update(exit_code=child.wait(),elapsed_seconds=time.monotonic()-started)
            if status['state']=='running':
                status['state']='completed' if status['exit_code']==0 else 'failed'
        finally:
            (args.work/'job_status.json').write_text(json.dumps(status,indent=2))
            try:
                backup(args,force=True)
            finally:
                # Leave watchdog running until the API acknowledges shutdown.
                stop_app(args)
                watchdog.terminate()


if __name__=='__main__':
    main()
