import gradio as gr
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=True)

def predict(prompt, audio_file_pth):

    tts.tts_to_file(text=prompt,
                file_path="output.wav",
                speaker_wav=audio_file_pth,
                language="en")

    return gr.make_waveform(audio="output.wav",)


title = "XTTS: MVP"

gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Prompt", info = "One or two sentences at a time is better* (max: 10)", placeholder = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",),
        gr.Audio(label="Upload Speaker WAV", type="filepath"),
    ],
    outputs=[
        gr.Video(label="Synthesised Speech"),
    ],
    title=title,
).launch(debug=True)