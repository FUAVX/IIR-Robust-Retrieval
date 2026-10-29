import json,pathlib
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
api=KaggleApi();api.authenticate()
ref='ndhoang2310/iir-b0-sfm100-cpu-preparation'
r=api.kernels_status(ref)
out=dict(status=str(r.status),failure_message=getattr(r,'failure_message',None))
print(json.dumps(out))
pathlib.Path(__file__).with_name('kernel-status.json').write_text(json.dumps(out,indent=2))
try:
 with api.build_kaggle_client() as client:
  req=ApiListKernelSessionOutputRequest();req.user_name='ndhoang2310';req.kernel_slug=ref.split('/')[1]
  result=client.kernels.kernels_api_client.list_kernel_session_output(req)
  log=result.log or ''
  pathlib.Path(__file__).with_name('kernel-run.log').write_text(log)
  if log:print(log[-3500:])
  print('Output files:',[x.file_name for x in result.files])
except Exception as error:
 print('Live outputs not available:',type(error).__name__)
