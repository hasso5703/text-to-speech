texte = [
"""
*century* n. A2 
The castle has changed very little over the centuries.
Le château a très peu changé au fil des siècles.
The history of this town spans four centuries.
L'histoire de cette ville couvre quatre siècles.
"""
]

import os
os.system('say "début"' )
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from gtts import gTTS
import soundfile as sf
import torch
import random
import string
import re
from pydub import AudioSegment
device = "cpu"
import cv2
import numpy as np
import textwrap
from PIL import ImageFont, ImageDraw, Image
from moviepy.editor import VideoFileClip, AudioFileClip, clips_array, CompositeVideoClip, concatenate_videoclips, concatenate_audioclips
from moviepy.config import change_settings

for i in range(len(texte)):
    print("weshhhh")
    texte_entree = texte[i]
    print("texte_entree : ", texte_entree)
    #if len(texte_entree) %2 != 0:
    #    print(texte_entree)
    #    os.system('say "erreur nombre de phrases impair, fin"' )
    #    break
    #-----------------------------------------
    word = "z_dont_know"
    mytext = []
    my_en_text = []
    #-----------------------------------------
    try:
        a = texte_entree.split("\n")
        if a[0] == "" or a[0] == ' ' or a[0] == '\n':
            del a[0]
        if a[-1] == "" or a[-1] == ' ' or a[-1] == '\n':
            del a[-1]
        w = a[0]
        del a[0]

        for i in range(len(a)):
            if "*" in a[i] or ":" in a[i]:
                a[i] = a[i][(a[i].index(":"))+1:].strip()
                a[i] = re.sub(r'^[^\w]+|[^\w]+$', '', a[i]).strip()
    except:
        print("ERROR!!!!!!")
        os.system('say "erreur dans une phrase"' )

    #---------------------------------------------
    frame_rate = 24
    def generate_video2(vduree, vtexte, vnom):
        largeur, hauteur = 1280, 720
        duree = vduree
        nom_video = vnom
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(nom_video, fourcc, frame_rate, (largeur, hauteur))
        texte = vtexte
        pil_font = ImageFont.truetype('/Library/Fonts/Arial.ttf', size=75)
        max_line_length = largeur - 20
        lines = textwrap.wrap(texte, width=30)
        line_height = int(pil_font.getmask(lines[0]).getbbox()[3] + 20)
        total_text_height = len(lines) * line_height
        position_x = (largeur - max_line_length) // 2
        position_y = (hauteur - total_text_height) // 2
        for temps_en_seconde in range(duree * frame_rate):
            image = np.zeros((hauteur, largeur, 3), dtype=np.uint8)
            y = position_y
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)
            for line in lines:
                text_mask = pil_font.getmask(line)
                text_size = (text_mask.getbbox()[2] - text_mask.getbbox()[0], text_mask.getbbox()[3] - text_mask.getbbox()[1])
                x = position_x + (max_line_length - text_size[0]) // 2
                draw.text((x, y), line, font=pil_font, fill=(255, 255, 255))
                y += line_height
            image = np.array(img_pil)
            video.write(image)
        video.release()
        cv2.destroyAllWindows()

    #-------------------------------------------------
    mot_extrait = re.search(r'\*(\w+)', w)
    if mot_extrait:
        mot = mot_extrait.group(1)
        print(mot.upper())
        word = mot
        print(word)
    if word=="z_dont_know":
        os.system('say "le mot n\'est pas reconnu"' )

    #---------------------------------------------------
    if len(a) % 2 != 0:
        print("ERRORR !!!!!")
        os.system('say "erreur nombre de phrases impair"' )

    #-----------------------------------------------------
    for i in range(len(a)):
        if i%2==0:
            my_en_text.append(a[i])
        else:
            mytext.append(a[i])

    #--------------------------------------------------------
    nb_text = len(my_en_text)
    #-----------------------------------------------------
    language = 'fr'
    i = 0
    for text in mytext:
        myobj = gTTS(text=text, lang=language, slow=False)
        i += 1
        myobj.save(f"temporaire/fr_audio_{i}.mp3")
        audio = AudioSegment.from_mp3(f"temporaire/fr_audio_{i}.mp3")
        silence = AudioSegment.silent(duration=200)
        audio_with_silence = silence + audio
        audio_accelere = audio_with_silence.speedup(playback_speed=1.1)
        audio_accelere.export(f"temporaire/fr_audio_{i}.mp3", format="mp3")

    #-----------------------------------------------------------------
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    #----------------------------------------------------------------------
    speakers = {
        'awb': 0,     # Scottish male
        'bdl': 1138,  # US male
        'clb': 2271,  # US female
        'jmk': 3403,  # Canadian male
        'ksp': 4535,  # Indian male
        'rms': 5667,  # US male
        'slt': 6799   # US female
    }
    #------------------------------------------------------
    def save_text_to_speech(text, nb, speaker=None):
        inputs = processor(text=text, return_tensors="pt").to(device)
        if speaker is not None:
            speaker_embeddings = torch.tensor(embeddings_dataset[speaker]["xvector"]).unsqueeze(0).to(device)
        else:
            speaker_embeddings = torch.randn((1, 512)).to(device)
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        if speaker is not None:
            output_filename = f"temporaire/en_audio_{nb}.mp3"
        else:
            random_str = ''.join(random.sample(string.ascii_letters+string.digits, k=5))
            output_filename = f"temporaire/en_audio_{nb}.mp3"
        sf.write(output_filename, speech.cpu().numpy(), samplerate=16000)
        return output_filename

    #------------------------------------------------------------
    for i,text in enumerate(my_en_text):
        voice = random.choice(['slt','bdl'])
        save_text_to_speech(text, i+1, speaker=speakers[voice])

    #-----------------------------------------------------------
    change_settings({"default_duration": 1/24})
    dossier_sortie = f"temporaire/words/{word}"
    base_path = f"temporaire/words/{word}/"
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)

    audio1 = AudioFileClip("temporaire/fr_audio_1.mp3")
    audio2 = AudioFileClip("temporaire/en_audio_1.mp3")
    audio_concatene = concatenate_audioclips([audio1, audio2])
    chemin_sortie = os.path.join(dossier_sortie, f"audio_{word}.mp3")
    audio_concatene.write_audiofile(chemin_sortie)
    audio = AudioSegment.from_file(f"temporaire/words/{word}/audio_{word}.mp3")
    duree_millisecondes = len(audio)
    duree_secondes = round(duree_millisecondes / 1000 + 0.15)
    if nb_text <= 1:
        generate_video2(duree_secondes, my_en_text[0], f"temporaire/words/{word}/video_en_test_final.mp4")
    else:
        generate_video2(duree_secondes, my_en_text[0], f"temporaire/words/{word}/video_en_test_1.mp4")
                                
    if nb_text > 1:
        fichier_concatene = audio_concatene  # Conservez le fichier concaténé précédent
        base_path = f"temporaire/words/{word}/"
        first_video_path = f"temporaire/words/{word}/video_en_test_1.mp4"
        for i in range(nb_text - 1):
            audio1 = AudioFileClip(f"temporaire/fr_audio_{i + 2}.mp3")
            audio2 = AudioFileClip(f"temporaire/en_audio_{i + 2}.mp3")
            fichier_concatene = concatenate_audioclips([fichier_concatene, audio1, audio2])
            chemin_sortie = os.path.join(dossier_sortie, f"audio_{word}.mp3")
            fichier_concatene.write_audiofile(chemin_sortie)
            audio_1 = AudioSegment.from_file(f"temporaire/fr_audio_{i + 2}.mp3")
            audio_2 = AudioSegment.from_file(f"temporaire/en_audio_{i + 2}.mp3")
            duree_millisecondes_ = len(audio_1) + len(audio_2)
            duree_secondes_ = round(duree_millisecondes_ / 1000 + 0.15)
            generate_video2(duree_secondes_, my_en_text[i+1], f"temporaire/words/{word}/video_en_test_{i+2}.mp4")
            clip1 = VideoFileClip(base_path+f"video_en_test_{i+1}.mp4")
            clip2 = VideoFileClip(base_path+f"video_en_test_{i+2}.mp4")
            final_clip = concatenate_videoclips([clip1, clip2])
            if i == nb_text-2:
                final_clip.write_videofile(base_path+f"video_en_test_final.mp4", codec="libx264", fps=frame_rate, audio_codec='aac')
            else:
                final_clip.write_videofile(base_path+f"video_en_test_{i+2}.mp4", codec="libx264", fps=frame_rate, audio_codec='aac')

    video_clip = VideoFileClip(base_path+f"video_en_test_final.mp4")
    audio_clip = AudioFileClip(base_path+f"audio_{word}.mp3")
    #---------------------------------------------------------------------------
    video_clip = video_clip.set_duration(audio_clip.duration)
    video_with_audio = video_clip.set_audio(audio_clip)
    video_with_audio.write_videofile(base_path+f"video_with_audio_{word}.mp4", codec="libx264", fps=frame_rate,  audio_codec='aac')
    #-----------------------------------------------------------------------------
    dossier = base_path # "temporaire/words/affect"
    for fichier in os.listdir(dossier):
        if "with" not in fichier:
            chemin_fichier = os.path.join(dossier, fichier)
            os.remove(chemin_fichier)

    #--------------------------------------------------------------
    for fichier in os.listdir("temporaire/"):
        if fichier.endswith(".mp3"):
            chemin_fichier = os.path.join("temporaire/", fichier)
            os.remove(chemin_fichier)

    #-------------------------------------------------------------
    message = f"fin pour {word}"
    os.system(f'say {message}')