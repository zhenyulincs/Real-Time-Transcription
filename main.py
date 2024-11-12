from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables")

from textual.app import App
from textual.widgets import Button, Static, ListView, ListItem, Log
from textual.containers import VerticalScroll
import openai
from textual.reactive import Reactive
import pyaudio
import wave
import threading
from faster_whisper import WhisperModel
import asyncio

import os
import numpy as np
import torch
import queue

# Variables to control recording
recording = False

# PyAudio settings
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Number of audio channels (single-channel for simplicity)
RATE = 16000  # Sample rate (16 kHz, compatible with Whisper)
CHUNK = 1024  # Buffer size for capturing audio data

# Initialize PyAudio
audio = pyaudio.PyAudio()
selected_device_index = None

# Initialize Whisper model
# model = whisper.load_model("turbo",device="cuda")  # Load the Whisper model (you can use "small" or "base" for speed)
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

# Queue for storing audio data chunks
audio_queue = queue.Queue()

# Function to list available audio devices
def list_audio_devices():
    device_list = []
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            device_list.append((i, device_info['name']))
    return device_list


class RecorderApp(App):
    recording_status: Reactive[str] = Reactive("Idle")
    transcribed_text: Reactive[str] = Reactive("")
    gpt_response: Reactive[str] = Reactive("")
    start_time = 0
    device_list = list_audio_devices()
    audio_file = "output.wav"
    transcript_file = "transcription.txt"
    last_processed_text = ""  # Track last processed text
    
    async def on_mount(self) -> None:
        # Create containers for transcript and response
        self.transcript_container = VerticalScroll(Static("Transcript:", id="transcript_label"))
        self.response_container = VerticalScroll(Static("ChatGPT Response:", id="response_label"))
        
        # Initialize widgets
        self.device_selection_view = ListView(
            *[ListItem(Static(f"{i}. {name}"), id=f"device_{i}") for i, name in self.device_list],
            name="device_list"
        )
        self.status = Static(f"Status: {self.recording_status}", name="status")
        self.transcript = Log(auto_scroll=True, name="transcript")
        self.gpt_response_widget = Log(auto_scroll=True, name="gpt_response")
        self.start_button = Button(label="Start Recording", name="start")
        self.stop_button = Button(label="Stop Recording", name="stop")

        # Disable buttons until device is selected
        self.start_button.disabled = True
        self.stop_button.disabled = True

        # Mount widgets
        await self.mount(Static("Select an Audio Device", id="title"))
        await self.mount(self.device_selection_view)
        await self.mount(self.status)
        await self.mount(self.transcript_container)
        await self.mount(self.transcript)
        await self.mount(self.response_container)
        await self.mount(self.gpt_response_widget)
        await self.mount(self.start_button, self.stop_button)

    async def on_list_view_selected(self, message: ListView.Selected) -> None:
        global selected_device_index
        selected_device_index = int(message.item.id.split("_")[1])
        self.start_button.disabled = False  # Enable the start button once a device is selected

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.name == "start":
            await self.start_recording()
        elif event.button.name == "stop":
            await self.stop_recording()

    async def start_recording(self) -> None:
        global recording
        if selected_device_index is not None and not recording:
            recording = True
            self.recording_status = "Recording..."
            self.start_button.disabled = True
            self.stop_button.disabled = False
            threading.Thread(target=self.record_audio).start()
            threading.Thread(target=self.transcribe_audio).start()

    async def stop_recording(self) -> None:
        global recording
        if recording:
            recording = False
            self.recording_status = "Stopped"
            self.start_button.disabled = False
            self.stop_button.disabled = True

            # Save both transcript and GPT response
            with open(self.transcript_file, "w") as f:
                f.write(self.transcribed_text)
            with open("gpt_response.txt", "w") as f:
                f.write(self.gpt_response)

            self.recording_status = f"Recording saved as {self.audio_file}, transcription and analysis saved"

    def record_audio(self) -> None:
        # Create a new wave file to write audio data to
        with wave.open(self.audio_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)

            # Open the audio stream
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                                rate=RATE, input=True, input_device_index=selected_device_index,
                                frames_per_buffer=CHUNK)

            while recording:
                data = stream.read(CHUNK)
                wf.writeframes(data)  # Write directly to the file to avoid excessive memory use
                audio_queue.put(data)  # Put the audio data into the queue for transcription

            # Stop and close the stream
            stream.stop_stream()
            stream.close()

    def transcribe_audio(self) -> None:
        full_audio = []
        has_new_content = False
        while recording or not audio_queue.empty():
            try:
                data = audio_queue.get(timeout=1)  # Get data from the queue
                full_audio.append(data)  # Collect audio data

                if len(full_audio) > 20:  # Wait until we have enough audio chunks
                    # Convert to numpy array
                    audio_data = np.concatenate([np.frombuffer(x, np.int16).astype(np.float32) / 32768.0 for x in full_audio])
                    try:
                        segments,_ = model.transcribe(audio_data, language="en", task="transcribe",
                                                        vad_filter=True,
                                                        beam_size=5,
                                                        word_timestamps=False,
                                                        vad_parameters=dict(min_silence_duration_ms=500))
                        
                        # Check if we got any segments
                        segment_list = list(segments)
                        if segment_list:  # Only process if we have segments
                            has_new_content = True
                            for segment in segment_list:
                                self.transcribed_text += "[%.2fs -> %.2fs] %s" % (self.start_time, self.start_time+segment.end, segment.text) + " \n"
                                self.start_time += segment.end
                            
                            # Only refresh if we have new content
                            if has_new_content:
                                self.refresh_transcription()
                                has_new_content = False
                    except Exception as e:
                        self.recording_status = f"Transcription error: {e}"
                    full_audio = []  # Clear buffer after transcription
            except queue.Empty:
                pass

    def refresh_transcription(self) -> None:
        self.transcript.write(self.transcribed_text)
        # Only call GPT if we have new content and enough text
        if (len(self.transcribed_text) > 50 and 
            self.transcribed_text != self.last_processed_text):
            self.last_processed_text = self.transcribed_text
            threading.Thread(target=self._async_gpt_call).start()

    def _async_gpt_call(self):
        async def _get_response():
            response = await self.get_gpt_response(self.transcribed_text)
            self.gpt_response = response
            self.gpt_response_widget.write(response)

        # Create new event loop for the thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_get_response())
        loop.close()

    async def get_gpt_response(self, text: str) -> str:
        try:
            # Initialize OpenAI 4th API key from environment
            client = openai.Client(api_key=OPENAI_API_KEY)
            
            # Send request to ChatGPT
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an interviewer and now you are interviewing with Google."},
                        {"role": "user", "content": f"Please Answer this interview question:\n{text}"}
                    ]
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting GPT response: {str(e)}"

    def watch_recording_status(self, recording_status: str) -> None:
        self.status.update(f"Status: {recording_status}")


if __name__ == "__main__":
    RecorderApp().run()  # Run the app as a class method, no 'self' needed

# Terminate PyAudio when app exits
audio.terminate()