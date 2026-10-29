import pathlib,time
from kaggle.api.kaggle_api_extended import KaggleApi
root=pathlib.Path(__file__).resolve().parent
ref='ndhoang2310/iir-b0-sfm100-cpu-preparation'
api=KaggleApi();api.authenticate()
seen=0
for attempt in range(10):
 position=0
 try:
  with (root/'kernel-live.log').open('a',buffering=1) as file:
   for event in api.kernels_logs_stream(ref):
    position+=1
    if position<=seen:continue
    seen=position
    value=event.get('data','');file.write(value)
    if any(marker in value for marker in ['Export:','API export:','PREPARATION COMPLETE:','archive_bytes','archive_sha256','[CELL FAILED]']):
     print(value.rstrip(),flush=True)
  if str(api.kernels_status(ref).status).endswith(('.COMPLETE','.ERROR')):break
 except Exception as error:
  print('Log stream reconnect:',type(error).__name__,flush=True)
 time.sleep(2)
