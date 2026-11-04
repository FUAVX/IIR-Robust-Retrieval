"""Run-to-completion controller for the user's explicit public-dataset request."""
import json,pathlib,subprocess,sys,time
from kaggle.api.kaggle_api_extended import KaggleApi
ROOT=pathlib.Path(__file__).resolve().parent
KERNEL='ndhoang2310/iir-b0-sfm100-cpu-preparation'
api=KaggleApi();api.authenticate()
started=time.monotonic();last_report=0
while True:
 response=api.kernels_status(KERNEL)
 status=str(response.status)
 (ROOT/'kernel-status.json').write_text(json.dumps(dict(status=status,failure_message=getattr(response,'failure_message','')),indent=2))
 now=time.monotonic()
 if now-last_report>=60:
  print('Kaggle preparation:',status,flush=True);last_report=now
 if status.endswith('.COMPLETE'):break
 if any(status.endswith('.'+s) for s in ['ERROR','CANCEL_REQUESTED','CANCELED','CANCELLED']):
  subprocess.run([sys.executable,str(ROOT/'status.py')],check=False)
  raise RuntimeError('Preparation did not complete: '+status)
 if now-started>90*60:raise TimeoutError('Preparation did not finish within this controller window.')
 time.sleep(30)
print('Preparation complete. Downloading output.',flush=True)
for attempt in range(3):
 result=subprocess.run([sys.executable,str(ROOT/'publish_prepared.py'),'download'])
 if result.returncode==0:break
 if attempt==2:raise RuntimeError('Output download failed after three resumable attempts.')
 time.sleep(10)
print('Output downloaded. Verifying and publishing public dataset.',flush=True)
subprocess.run([sys.executable,str(ROOT/'publish_prepared.py'),'publish'],check=True)
print('Dataset upload submitted. Waiting for processing and public verification.',flush=True)
for attempt in range(120):
 result=subprocess.run([sys.executable,str(ROOT/'verify_public.py')])
 if result.returncode==0:
  print('PUBLIC DATASET VERIFIED: https://www.kaggle.com/datasets/ndhoang2310/iir-b0-sfm100-prepared',flush=True)
  break
 if result.returncode!=2:raise RuntimeError('Public dataset verification failed; inspect output before proceeding.')
 time.sleep(30)
else:raise TimeoutError('Dataset processing has not completed within one hour.')
