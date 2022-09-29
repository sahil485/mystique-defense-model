import whisper
from record import RecAUD

gui = RecAUD()
gui.open()

model = whisper.load_model("base")
result = model.transcribe("test_recording.wav")
print(result["text"])