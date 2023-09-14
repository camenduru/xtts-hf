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

description = """
<p>For faster inference without waiting in the queue, you should duplicate this space and upgrade to GPU via the settings.
<br/>
<a href="https://huggingface.co/spaces/coqui/xtts?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
</p>
XTTS is a Voice generation model that lets you clone voices into different languages by using just a quick 3-second audio clip. 
Built on Tortoise, XTTS has important model changes that make cross-language voice cloning and multi-lingual speech generation super easy. 
<br/>
This is the same model that powers Coqui Studio, and Coqui API, however we apply a few tricks to make it faster and support streaming inference.
"""

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
    description=description,
).launch(debug=True)
