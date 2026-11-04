import json,pathlib,requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
root=pathlib.Path(__file__).resolve().parent
api=KaggleApi();api.authenticate()
ref='ndhoang2310/iir-b0-cpu-input-probe'
status=str(api.kernels_status(ref).status);print('Probe:',status,flush=True)
if status.endswith(('.COMPLETE','.ERROR')):
 with api.build_kaggle_client() as client:
  request=ApiListKernelSessionOutputRequest();request.user_name='ndhoang2310';request.kernel_slug=ref.split('/')[1]
  output=client.kernels.kernels_api_client.list_kernel_session_output(request)
  if output.log:print(output.log[-1500:])
  for file in output.files:
   if file.file_name=='input_probe.json':
    response=requests.get(file.url,timeout=(30,120));response.raise_for_status();data=response.json()
    (root/'input-probe-result.json').write_text(json.dumps(data,indent=2))
    print(json.dumps({k:v for k,v in data.items() if not k.endswith('_samples')},indent=2))
