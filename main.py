from textual.app import App
from textual.widgets import Button, Static, ListView, ListItem
from textual.reactive import Reactive
import pyaudio
import wave
import threading
from faster_whisper import WhisperModel

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
    start_time = 0
    device_list = list_audio_devices()
    audio_file = "output.wav"
    transcript_file = "transcription.txt"
    
    async def on_mount(self) -> None:
        # Show available audio devices
        self.device_selection_view = ListView(
            *[ListItem(Static(f"{i}. {name}"), id=f"device_{i}") for i, name in self.device_list],
            name="device_list"
        )
        self.status = Static(f"Status: {self.recording_status}", name="status")
        self.transcript = Static(self.transcribed_text, name="transcript")
        self.start_button = Button(label="Start Recording", name="start")
        self.stop_button = Button(label="Stop Recording", name="stop")

        # Disable the buttons until a device is selected
        self.start_button.disabled = True
        self.stop_button.disabled = True

        # Mount the widgets to the layout
        await self.mount(Static("Select an Audio Device", id="title"))
        await self.mount(self.device_selection_view)
        await self.mount(self.status)
        await self.mount(self.transcript)
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

            # # Wait for all audio data to be transcribed
            # while not audio_queue.empty():
            #     pass

            # Save the transcription to a text file
            with open(self.transcript_file, "w") as f:
                f.write(self.transcribed_text)

            # Update UI status
            self.recording_status = f"Recording saved as {self.audio_file}, transcription saved as {self.transcript_file}"

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
                        for segment in segments:
                            self.transcribed_text += "[%.2fs -> %.2fs] %s" % (self.start_time, self.start_time+segment.end, segment.text) + " \n"
                            self.start_time += segment.end
                        self.refresh_transcription()
                    except Exception as e:
                        self.recording_status = f"Transcription error: {e}"
                    full_audio = []  # Clear buffer after transcription
            except queue.Empty:
                pass

    def refresh_transcription(self) -> None:
        self.transcript.update(self.transcribed_text)

    def watch_recording_status(self, recording_status: str) -> None:
        self.status.update(f"Status: {recording_status}")


if __name__ == "__main__":
    RecorderApp().run()  # Run the app as a class method, no 'self' needed

# Terminate PyAudio when app exits
audio.terminate()