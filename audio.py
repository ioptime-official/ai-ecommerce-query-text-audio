import boto3
import time

def synthesize_speech(text, voice_id, output_format='mp3'):
    # Create a Polly client
    polly_client = boto3.client('polly')

    # Synthesize speech
    response = polly_client.synthesize_speech(
        OutputFormat=output_format,
        Text=text,
        VoiceId=voice_id
    )

    return response['AudioStream'].read()

def upload_to_s3(audio_data, bucket_name, object_key):
    # Create an S3 client
    s3_client = boto3.client('s3')

    # Upload the audio file to S3
    s3_client.put_object(
        Body=audio_data,
        Bucket=bucket_name,
        Key=object_key
    )

def main():
    # Specify the text to be synthesized
    text_to_synthesize = "Hello, this is a sample text to be converted to speech."

    # Specify the voice ID (e.g., 'Joanna' for English (US))
    voice_id = 'Joanna'

    # Specify the S3 bucket name and object key
    s3_bucket_name = 'audioali'
    object_key = 'audio/sample.mp3'

    # Synthesize speech
    audio_data = synthesize_speech(text_to_synthesize, voice_id)

    # Upload the audio file to S3
    upload_to_s3(audio_data, s3_bucket_name, object_key)

    print(f"Speech synthesized and uploaded to S3://{s3_bucket_name}/{object_key}")

if __name__ == "__main__":
    main()
