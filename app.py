import gradio as gr
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
tts.to("cuda")


def predict(prompt, language, audio_file_pth):
    tts.tts_to_file(
        text=prompt,
        file_path="output.wav",
        speaker_wav=audio_file_pth,
        language=language,
    )

    return gr.make_waveform(
        audio="output.wav",
    )


title = "Coqui🐸 XTTS"

gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="Text Prompt",
            info="One or two sentences at a time is better",
            placeholder="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
        ),
        gr.Dropdown(
            label="Language",
            info="Select an output language for the synthesised speech",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cz",
                "ar",
                "zh",
            ],
            max_choices=1,
            value="en"
        ),
        gr.Audio(
            label="Reference Audio",
            info="Upload a reference audio for target speaker voice",
            type="filepath",
            value="examples/en_speaker_6.wav"
        ),
    ],
    outputs=[
        gr.Video(label="Synthesised Speech"),
    ],
    title=title,
).launch(debug=True)
