import os
import json
import boto3
import requests

s3 = boto3.client('s3')

HF_API_TOKEN = os.environ['HF_API_TOKEN']

MODELS = {
    "mobilenet": "google/mobilenet_v1_0.75_192",
    "resnet": "microsoft/resnet-50",
    "mitb0": "nvidia/mit-b0",
    "yolos": "hustvl/yolos-tiny"
}

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

def call_hf_api(model_id, image_bytes):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(url, headers=HEADERS, data=image_bytes)
    response.raise_for_status()
    return response.json()

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # ჩამოტვირთე სურათი S3-დან
    s3_response = s3.get_object(Bucket=bucket, Key=key)
    image_bytes = s3_response['Body'].read()

    image_name = key.split('/')[-1].split('.')[0]

    results = {}


    for short_name, model_id in MODELS.items():
        try:
            result = call_hf_api(model_id, image_bytes)
            results[short_name] = result

            
            json_key = f"json/{short_name}_{image_name}.json"
            s3.put_object(
                Bucket=bucket,
                Key=json_key,
                Body=json.dumps(result, indent=2),
                ContentType='application/json'
            )
        except Exception as e:
            results[short_name] = {"error": str(e)}

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Processed image with multiple models",
            "results": results
        })
    }
