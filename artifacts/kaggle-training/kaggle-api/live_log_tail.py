import json,pathlib,signal,time
from kaggle.api.kaggle_api_extended import KaggleApi
root=pathlib.Path(__file__).resolve().parent
api=KaggleApi();api.authenticate()
texts=[];events=[]
deadline=time.monotonic()+12
def stop(*args):raise TimeoutError('Live-log snapshot deadline')
signal.signal(signal.SIGALRM,stop);signal.alarm(12)
try:
 for event in api.kernels_logs_stream('ndhoang2310/iir-b0-sfm100-cpu-preparation'):
  texts.append(event.get('data',''));events.append(event)
except Exception:
 if time.monotonic()<deadline-1:
  raise
finally:
 signal.alarm(0)
 (root/'live-log-snapshot.json').write_text(json.dumps(events))
 (root/'live-log-snapshot.txt').write_text(''.join(texts))
 print('Events captured:',len(events))
 print(''.join(texts)[-2000:])
