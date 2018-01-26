# 네이버 음성합성 Open API 예제
# import os
import sys
import urllib.request

def TextToKorSpeechNAVER(Client_Id, Client_Secret, str, filename, extension='mp3'):
  client_id = Client_Id
  client_secret = Client_Secret
  encText = urllib.parse.quote(str)
  data = "speaker=mijin&speed=0&text=" + encText;
  url = "https://openapi.naver.com/v1/voice/tts.bin"
  request = urllib.request.Request(url)
  request.add_header("X-Naver-Client-Id", client_id)
  request.add_header("X-Naver-Client-Secret", client_secret)
  response = urllib.request.urlopen(request, data=data.encode('utf-8'))
  rescode = response.getcode()
  if (rescode == 200):
    print("TTS mp3 저장")
    response_body = response.read()
    with open(filename+'.'+extension, 'wb') as f:
      f.write(response_body)
  else:
    print("Error Code:" + rescode)


#sudo pip install pyttsx - sounds like robotic
'''import pyttsx

engine = pyttsx.init()
engine.say('Hello World')
engine.runAndWait()'''

#sudo apt-get install espeak
'''import os
os.system("espeak 'Hello world'")'''

#sudo pip install gTTS
#sudo apt-get install mpg321
from gtts import gTTS
import os


def TextToKorSpeechGTTS(str, filename, extension='mp3'):
  tts = gTTS(text=str, lang='ko', slow=False)
  try:
    tts.save(filename+'.'+extension)
  except PermissionError as pe:
    str(pe)
    return 0
  except Exception as e:
    str(e)
    return 0
  else:
    print("Creating File completed")
    return 1

#print(TextToKorSpeech('안녕하세요', '5', 'wav'))
#os.system("mpg321 hello.mp3")
