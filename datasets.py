from urllib.parse import urlparse
import pandas as pd
import yaml
import shutil
import os
import json
from glob import glob
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

dataset_path = os.getenv("DATASET_PATH")


def Clotho():
    clotho_dev = pd.read_csv(f'{dataset_path}/clotho/clotho_captions_development.csv')
    clotho_dev["split"] = "train"
    clotho_eval = pd.read_csv(f'{dataset_path}/clotho/clotho_captions_evaluation.csv')
    clotho_eval["split"] = "valid"
    clotho_val = pd.read_csv(f'{dataset_path}/clotho/clotho_captions_validation.csv')
    clotho_val["split"] = "test"

    clotho = pd.concat([clotho_dev, clotho_eval, clotho_val], ignore_index=True)
    clotho = pd.melt(clotho, id_vars=['file_name', 'split'], value_vars=['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5'],
                var_name='caption_num', value_name='caption')
    clotho['caption_num'] = clotho['caption_num'].str.extract('(\d+)', expand=False).astype(int)
    clotho = clotho[["file_name", "caption", "split"]]
    return clotho


def AudioCaps():
    train = pd.read_csv(f"{dataset_path}/AudioCaps/train.csv")
    train["split"] = "train"
    val = pd.read_csv(f"{dataset_path}/AudioCaps/val.csv")
    val["split"] = "valid"
    test = pd.read_csv(f"{dataset_path}/AudioCaps/test.csv")
    test["split"] = "test"

    audiocaps = pd.concat([train, val, test], ignore_index=True)
    audiocaps["file_name"] = audiocaps["youtube_id"].apply(lambda x: f"Y{x}.wav")
    audiocaps = audiocaps[["file_name", "caption", "split"]]

    return audiocaps


def MACS():
    with open(f'{dataset_path}/macs/MACS.yaml', 'r') as file:
        yaml_data = yaml.safe_load(file)

    df_data = []
    for item in yaml_data["files"]:
        filename = item["filename"]
        for annotation in item["annotations"]:
            df_data.append({"file_name": filename, "caption": annotation["sentence"]})
    df = pd.DataFrame(df_data)
    df["split"] = "train"
    return df

def WavText5K():
    df = pd.read_csv(f"{dataset_path}/WavText5K/WavText5K.csv")
    df.rename(columns={"description": "caption", "fname": "file_name"}, inplace=True)
    df["split"] = "train"
    df = df[["file_name", "caption", "split"]]
    return df

def AutoACD():
    train = pd.read_csv(f"{dataset_path}/AutoACD/train.csv")
    train["split"] = "train"
    test = pd.read_csv(f"{dataset_path}/AutoACD/test.csv")
    test["split"] = "test"
    data = pd.concat([train, test], ignore_index=True)
    data["file_name"] = data["youtube_id"] + ".flac"
    data.drop(columns=["youtube_id"], inplace=True)
    return data

def AudioCaption():
    hospital = pd.read_json(f"{dataset_path}/AudioCaption/hospital_en_all.json")
    hospital['caption_index'] = hospital.groupby('filename').cumcount() + 1
    hospital_zh_dev = pd.read_json(f"{dataset_path}/AudioCaption/hospital_zh_dev.json")

    # match on filename and caption_index
    hospital["split"] = hospital.apply(lambda x: "train" if x["filename"] in hospital_zh_dev["filename"].values else "test", axis=1)

    hospital["filename"] = hospital["filename"].str.replace("hospital_3707", "Hospital")
    hospital.rename(columns={"filename": "file_name"}, inplace=True)

    car_dev = pd.read_json(f"{dataset_path}/AudioCaption/car_en_dev.json")
    car_dev["split"] = "train"
    car_eval = pd.read_json(f"{dataset_path}/AudioCaption/car_en_eval.json")
    car_eval["split"] = "test"
    car = pd.concat([car_dev, car_eval], ignore_index=True)
    car["filename"] = car["filename"].str.replace("car_3610", "Car")
    car.rename(columns={"filename": "file_name"}, inplace=True)
    data = pd.concat([hospital, car], ignore_index=True)
    data = data[["file_name", "caption", "split"]]
    return data

def SoundDescs():
    descriptions = pd.read_pickle(f"{dataset_path}/SoundDescs/descriptions.pkl")
    descriptions = pd.DataFrame.from_dict(descriptions, orient="index", columns=["description"])
    categories = pd.read_pickle(f"{dataset_path}/SoundDescs/categories.pkl")
    categories = pd.DataFrame.from_dict(categories, orient="index", columns=["category1", "category2", "category3"])
    extra_info = pd.read_pickle(f"{dataset_path}/SoundDescs/extra_info.pkl")
    extra_info = pd.DataFrame.from_dict(extra_info, orient="index", columns=["extra_info"])

    train = pd.read_csv(f"{dataset_path}/SoundDescs/train_list.txt", header=None, names=["id"])
    train["split"] = "train"
    val = pd.read_csv(f"{dataset_path}/SoundDescs/val_list.txt", header=None, names=["id"])
    val["split"] = "valid"
    test = pd.read_csv(f"{dataset_path}/SoundDescs/test_list.txt", header=None, names=["id"])
    test["split"] = "test"
    sounddescs = pd.concat([train, val, test], ignore_index=True)
    sounddescs = pd.merge(sounddescs, descriptions, left_on="id", right_index=True)
    sounddescs = pd.merge(sounddescs, categories, left_on="id", right_index=True)
    sounddescs = pd.merge(sounddescs, extra_info, left_on="id", right_index=True)
    sounddescs.rename(columns={"id": "file_name", "description": "caption"}, inplace=True)

    sounddescs["file_name"] = sounddescs["file_name"].str.upper()
    sounddescs["file_name"] = sounddescs["file_name"].apply(lambda x: f"{x}.wav")

    sounddescs = sounddescs[["file_name", "caption", "split"]]

    return sounddescs

def WavCaps(): # TODO: file_name not yet completely matches name of the files 
    with open(f"{dataset_path}/WavCaps/as_final.json", "r") as file:
        audioset = json.load(file)
        audioset = pd.DataFrame(audioset["data"])
        audioset.rename(columns={"id": "file_name"}, inplace=True)
        audioset.drop(columns=["audio"], inplace=True)
        audioset["type"] = "audioset"
    with open(f"{dataset_path}/WavCaps/bbc_final.json", "r") as file:
        bbc = json.load(file)
        bbc = pd.DataFrame(bbc["data"])
        bbc.rename(columns={"id": "file_name"}, inplace=True)
        bbc.drop(columns=["description", "category", "audio", "download_link"], inplace=True)
        bbc["type"] = "bbc"
    with open(f"{dataset_path}/WavCaps/fsd_final.json", "r") as file:
        fsd = json.load(file)
        fsd = pd.DataFrame(fsd["data"])
        fsd.drop(columns=["id", "href", "tags", "author", "description", "audio", "download_link"], inplace=True)
        fsd["type"] = "fsd"
    with open(f"{dataset_path}/WavCaps/sb_final.json", "r") as file:
        soundbible = json.load(file)
        soundbible = pd.DataFrame(soundbible["data"])
        soundbible.rename(columns={"id": "file_name"}, inplace=True)
        soundbible.drop(columns=["audio", "download_link", "href", "title", "author", "description"], inplace=True)
        soundbible["type"] = "soundbible"
    wavcaps = pd.concat([audioset, bbc, fsd, soundbible], ignore_index=True)
    wavcaps["split"] = "train"
    wavcaps = wavcaps[["file_name", "caption", "split", "type"]] 
    return wavcaps

def TextToAudioGrounding():
    train = pd.read_json(f"{dataset_path}/TextToAudioGrounding/train.json")
    train["split"] = "train"
    val = pd.read_json(f"{dataset_path}/TextToAudioGrounding/val.json")
    val["split"] = "valid"
    test = pd.read_json(f"{dataset_path}/TextToAudioGrounding/test.json")
    test["split"] = "test"
    combined = pd.concat([train, val, test], ignore_index=True)
    combined.rename(columns={"tokens": "caption", "audio_id": "file_name"}, inplace=True)
    combined = combined[["file_name", "caption", "split"]]
    return combined

def FAVDBench():
    df = pd.read_csv(f"{dataset_path}/FAVDBench/FAVDBench_Audio_Updated.csv")
    
    return df

def ClothoDetail():
    with open(f'{dataset_path}/ClothoDetail/Clotho-detail-annotation.json', 'r') as file:
        clotho_detail = json.load(file)
    clotho_detail = pd.DataFrame(clotho_detail["annotations"])
    clotho_dev = pd.read_csv(f'{dataset_path}/clotho/clotho_captions_development.csv')
    clotho_eval = pd.read_csv(f'{dataset_path}/clotho/clotho_captions_evaluation.csv')
    # clotho_val = pd.read_csv(f'{dataset_path}/clotho/clotho_captions_validation.csv')
    clotho_detail["audio_id"] = clotho_detail["audio_id"] + ".wav"
    clotho_dev["split"] = "train"
    clotho_eval["split"] = "valid"

    clotho = pd.concat([clotho_dev, clotho_eval], ignore_index=True)

    combined = pd.merge(clotho_detail, clotho, left_on="audio_id", right_on="file_name", how="left")
    combined = combined[["file_name", "caption", "split"]]
    
    # this item will get merged in both train and valid, so we remove it from train
    drop_index = combined.loc[(combined["file_name"] == "FREEZER_DOOR_OPEN_CLOSE.wav") & (combined["split"] == "dev")].index
    combined.drop(drop_index, inplace=True)

    return combined

def VGGSound():
    df = pd.read_csv(f"{dataset_path}/VGGSound/vggsound.csv", header=None, names=["file_name", "start_sec", "caption", "split"])
    # pad up start_sec column to 6 digits with 0
    df["file_name"] = df.apply(lambda x: f"{x['file_name']}_{str(x['start_sec']).zfill(6)}.mp4", axis=1)
    df.drop(columns=["start_sec"], inplace=True)
    return df

def SoundingEarth():
    df = pd.read_csv(f"{dataset_path}/SoundingEarth/metadata.csv")
    df = df[["file_name", "caption", "split"]]
    # Drop rows with null values
    df = df.dropna()
    return df

def FSD50k():
    dev_data = pd.read_csv(f"{dataset_path}/FSD50k/FSD50K.ground_truth/dev.csv")
    dev_data["split"] = "train"
    dev_data["file_name"] = dev_data["fname"].apply(lambda x: f"FSD50K.dev_audio/{x}.wav")
    eval_data = pd.read_csv(f"{dataset_path}/FSD50k/FSD50K.ground_truth/eval.csv")
    eval_data["split"] = "test"
    eval_data["file_name"] = dev_data["fname"].apply(lambda x: f"FSD50K.eval_audio/{x}.wav")
    combined = pd.concat([dev_data, eval_data], ignore_index=True)
    
    metadata_dev = pd.read_json(f"{dataset_path}/FSD50k/FSD50K.metadata/dev_clips_info_FSD50K.json", orient="index")
    metadata_eval = pd.read_json(f"{dataset_path}/FSD50k/FSD50K.metadata/eval_clips_info_FSD50K.json", orient="index")

    metadata = pd.concat([metadata_dev, metadata_eval])
    combined_result = pd.merge(combined, metadata, left_on="fname", right_index=True)
    def restructure(row):
        splitted = row["title"].split(".")
        if len(splitted) > 1:
            row["title"] = splitted[0]
        return f'{row["title"]}. {row["description"]}'

    combined_result["caption"] = combined_result.apply(lambda x: restructure(x), axis=1)
    combined_result["file_name"] = combined_result["file_name"].str.replace("FSD50K.dev_audio/", "")
    combined_result["file_name"] = combined_result["file_name"].str.replace("FSD50K.eval_audio/", "")
    combined_result = combined_result[["file_name", "caption", "split"]]
    return combined_result

def Audioset():
    # Load the datasets
    balanced_train = pd.read_csv(f'{dataset_path}/Audioset/balanced_train_segments.csv')
    unbalanced_train = pd.read_csv(f'{dataset_path}/Audioset/unbalanced_train_segments.csv')
    eval_segments = pd.read_csv(f'{dataset_path}/Audioset/eval_segments.csv')
    class_labels_indices = pd.read_csv(f'{dataset_path}/Audioset/class_labels_indices.csv')
    balanced_train["split"] = "balanced"
    unbalanced_train["split"] = "unbalanced"
    eval_segments["split"] = "test"

    # Assuming the class_labels_indices has columns 'mid' and 'display_name' for mapping
    label_mapping = class_labels_indices.set_index('mid')['display_name'].to_dict()

    def replace_data(row):
        row = row.lstrip('"')
        row = row.rstrip('"')
        row = row.split(",")
        return ", ".join([label_mapping[x] for x in row])
    
    all_data = pd.concat([balanced_train, unbalanced_train, eval_segments], ignore_index=True)

    # Replace the labels with the display names
    all_data['caption'] = all_data['LabelIDs'].apply(replace_data)

    
    all_data["file_name"] = all_data["YouTubeID"].apply(lambda x: f"Y{x}.wav")
    all_data = all_data[["file_name", "caption", "split"]]
    return all_data

def ClothoAQA():
    train = pd.read_csv(f"{dataset_path}/ClothoAQA/clotho_aqa_train.csv")
    train["split"] = "train"
    val = pd.read_csv(f"{dataset_path}/ClothoAQA/clotho_aqa_val.csv")
    val["split"] = "valid"
    test = pd.read_csv(f"{dataset_path}/ClothoAQA/clotho_aqa_test.csv")
    test["split"] = "test"
    combined = pd.concat([train, val, test], ignore_index=True)
    # metadata = pd.read_csv(f"{dataset_path}/ClothoAQA/clotho_aqa_metadata.csv")
    # combined = pd.merge(combined, metadata, left_on="audio_id", right_on="file_name")
    combined["caption"] = combined.apply(lambda x: f'{x["QuestionText"]} {x["answer"]}', axis=1)
    combined = combined[["file_name", "caption", "split"]]
    return combined

def ClothoV2GPT():
    df = pd.read_json(f"{dataset_path}/ClothoV2GPT/variations.json")
    df["path"] = df["path"].str.replace("/home/paul/shared/clotho_v2/development/", "")
    # Explode the 'variations' column to create separate rows for each variation
    df = df.explode('variations')

    df.drop(columns=["caption"], inplace=True)
    
    # Rename columns for clarity
    df = df.rename(columns={'path': 'file_name', 'caption': 'old_caption', 'variations': 'caption'})
    
    # Reset index to ensure proper alignment
    df = df.reset_index(drop=True)
    
    # Select only the required columns
    df = df[['file_name', 'caption']]
    df["split"] = "train"
    return df

def AudioEgoVLP():
    ego4d = pd.read_csv(f"{dataset_path}/AudioEgoVLP/ego4d_silence_low_high_gpt_extracted_audio.csv")
    ego4d["id"] = ego4d["id"].astype(str)
    ego4d["file_name"] = ego4d["video_uid"] + "_" + ego4d["id"] + ".flac"
    ego4d.rename(columns={"clip_text": "caption"}, inplace=True)
    ego4d = ego4d[["file_name", "caption"]]

    epic = pd.read_csv(f"{dataset_path}/AudioEgoVLP/EPIC_100_retrieval_test_gptfiltered_high_gpt.csv")
    epic["file_name"] = epic["narration_id"] + ".flac"
    epic.rename(columns={"narration": "caption"}, inplace=True)
    epic = epic[["file_name", "caption"]]

    combined = pd.concat([ego4d, epic], ignore_index=True)
    combined["split"] = "train"
    return combined

def AFAudioSet():
    df = pd.read_json(f"{dataset_path}/AFAudioSet/AF-AudioSet.json", lines=True)
    df["file_name"] = "Y" +df["ytid"] + ".wav"
    df["split"] = "train"
    df = df[["file_name", "caption", "split"]]
    return df

def SoundVECaps():
    df = pd.read_csv(f"{dataset_path}/SoundVECaps/Sound-VECaps_audio.csv")
    df.rename(columns={"id": "file_name"}, inplace=True)
    df["split"] = "train"
    df = df[["file_name", "caption", "split"]]
    return df


def CAPTDURE():
    single_source_caption = pd.read_csv(f"{dataset_path}/CAPTDURE/single_source_caption.csv", sep="\t")
    mixture_source_caption = pd.read_csv(f"{dataset_path}/CAPTDURE/mixture_source_caption.csv", sep="\t")

    single_source_caption["subdirectory"] = "single_source_source"
    mixture_source_caption["subdirectory"] = "multiple_source_sound"

    single_source_caption["split"] = "train"
    mixture_source_caption["split"] = "train"

    combined = pd.concat([single_source_caption, mixture_source_caption], ignore_index=True)
    combined.rename(columns={"wavfile": "file_name", "caption (en)": "caption"}, inplace=True)
    combined["file_name"] = combined.apply(lambda x: x["subdirectory"] + "/" + x["file_name"].split("/")[-1], axis=1)
    combined = combined[["file_name", "caption", "split"]]
    return combined

def AnimalSpeak():
    data = pd.read_csv(f"{dataset_path}/AnimalSpeak/AnimalSpeak_correct.csv")
    # data.rename(columns={"caption": "caption1"}, inplace=True)
    # data = pd.melt(data, id_vars=['i', 'url', 'source', 'recordist', 'species_common', 'species_scientific'], 
    #                 value_vars=['caption1', 'caption2'], 
    #                 var_name='caption_type', value_name='caption')
    # data["split"] = "train"
    # data = data[["url", "caption", "split"]]
    # data["file_name"] = data["url"].apply(lambda x: f"{x.split('/')[-1]}")
    # data["file_name"] = data.apply(lambda x: str(x.name) + os.path.splitext(os.path.basename(urlparse(x['url']).path))[1], axis=1)

    data = data[["file_name", "caption", "split"]]
    return data

def mAQA():
    binary_test_french = pd.read_csv(f"{dataset_path}/mAQA/binary_test_french.csv")
    binary_test_french["split"] = "test"
    binary_test_italian = pd.read_csv(f"{dataset_path}/mAQA/binary_test_italian.csv")
    binary_test_italian["split"] = "test"
    binary_train_dutch = pd.read_csv(f"{dataset_path}/mAQA/binary_train_dutch.csv")
    binary_train_dutch["split"] = "train"
    binary_train_german = pd.read_csv(f"{dataset_path}/mAQA/binary_train_german.csv")
    binary_train_german["split"] = "train"
    binary_train_portuguese = pd.read_csv(f"{dataset_path}/mAQA/binary_train_portuguese.csv")
    binary_train_portuguese["split"] = "train"
    binary_val_eng = pd.read_csv(f"{dataset_path}/mAQA/binary_val_eng.csv")
    binary_val_eng["split"] = "valid"
    binary_val_hindi = pd.read_csv(f"{dataset_path}/mAQA/binary_val_hindi.csv")
    binary_val_hindi["split"] = "valid"
    binary_val_spanish = pd.read_csv(f"{dataset_path}/mAQA/binary_val_spanish.csv")
    binary_val_spanish["split"] = "valid"
    single_word_train = pd.read_csv(f"{dataset_path}/mAQA/single_word_train.csv")
    single_word_train["split"] = "train"
    binary_test_dutch = pd.read_csv(f"{dataset_path}/mAQA/binary_test_dutch.csv")
    binary_test_dutch["split"] = "test"
    binary_test_german = pd.read_csv(f"{dataset_path}/mAQA/binary_test_german.csv")
    binary_test_german["split"] = "test"
    binary_test_portuguese = pd.read_csv(f"{dataset_path}/mAQA/binary_test_portuguese.csv")
    binary_test_portuguese["split"] = "test"
    binary_train_eng = pd.read_csv(f"{dataset_path}/mAQA/binary_train_eng.csv")
    binary_train_eng["split"] = "train"
    binary_train_hindi = pd.read_csv(f"{dataset_path}/mAQA/binary_train_hindi.csv")
    binary_train_hindi["split"] = "train"
    binary_train_spanish = pd.read_csv(f"{dataset_path}/mAQA/binary_train_spanish.csv")
    binary_train_spanish["split"] = "train"
    binary_val_french = pd.read_csv(f"{dataset_path}/mAQA/binary_val_french.csv")
    binary_val_french["split"] = "valid"
    binary_val_italian = pd.read_csv(f"{dataset_path}/mAQA/binary_val_italian.csv")
    binary_val_italian["split"] = "valid"
    single_word_val = pd.read_csv(f"{dataset_path}/mAQA/single_word_val.csv")
    single_word_val["split"] = "valid"
    binary_test_eng = pd.read_csv(f"{dataset_path}/mAQA/binary_test_eng.csv")
    binary_test_eng["split"] = "test"
    binary_test_hindi = pd.read_csv(f"{dataset_path}/mAQA/binary_test_hindi.csv")
    binary_test_hindi["split"] = "test"
    binary_test_spanish = pd.read_csv(f"{dataset_path}/mAQA/binary_test_spanish.csv")
    binary_test_spanish["split"] = "test"
    binary_train_french = pd.read_csv(f"{dataset_path}/mAQA/binary_train_french.csv")
    binary_train_french["split"] = "train"
    binary_train_italian = pd.read_csv(f"{dataset_path}/mAQA/binary_train_italian.csv")
    binary_train_italian["split"] = "train"
    binary_val_dutch = pd.read_csv(f"{dataset_path}/mAQA/binary_val_dutch.csv")
    binary_val_dutch["split"] = "valid"
    binary_val_german = pd.read_csv(f"{dataset_path}/mAQA/binary_val_german.csv")
    binary_val_german["split"] = "valid"
    binary_val_portuguese = pd.read_csv(f"{dataset_path}/mAQA/binary_val_portuguese.csv")
    binary_val_portuguese["split"] = "valid"
    single_word_test = pd.read_csv(f"{dataset_path}/mAQA/single_word_test.csv")
    single_word_test["split"] = "test"

    all_together = pd.concat([binary_test_french, binary_test_italian, binary_train_dutch, binary_train_german, binary_train_portuguese, binary_val_eng, binary_val_hindi, binary_val_spanish, single_word_train, binary_test_dutch, binary_test_german, binary_test_portuguese, binary_train_eng, binary_train_hindi, binary_train_spanish, binary_val_french, binary_val_italian, single_word_val, binary_test_eng, binary_test_hindi, binary_test_spanish, binary_train_french, binary_train_italian, binary_val_dutch, binary_val_german, binary_val_portuguese, single_word_test])

    all_together["caption"] = all_together.apply(lambda x: f"{x['QuestionText']} {x['answer']}", axis=1)
    all_together = all_together[["file_name", "caption", "split"]]
    return all_together


def DAQA():
    with open(f"{dataset_path}/DAQA/daqa_train_questions_answers_5.json", "r") as file:
        train = json.load(file)
    with open(f"{dataset_path}/DAQA/daqa_val_questions_answers.json", "r") as file:
        val = json.load(file)
    with open(f"{dataset_path}/DAQA/daqa_test_questions_answers.json", "r") as file:
        test = json.load(file)
    
    train_df = pd.DataFrame(train["questions"])
    val_df = pd.DataFrame(val["questions"])
    test_df = pd.DataFrame(test["questions"])

    val_df["set"] = "valid"

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined["caption"] = combined.apply(lambda x: f"{x['question']} {x['answer']}", axis=1)
    combined.rename(columns={"set": "split", "audio_filename": "file_name"}, inplace=True)
    combined = combined[["file_name", "caption", "split"]]
    return combined

def MULTIS():
    def extract_human(conversation):
        if len(conversation) > 1:
            return list(filter(lambda x: x["from"] == "human", conversation))[0]["value"].replace("<audio>", "").replace("\n", "")
        else:
            return None
    def extract_gpt(conversation):
        if len(conversation) > 1:
            return list(filter(lambda x: x["from"] == "gpt", conversation))[0]["value"]
        else:
            return list(filter(lambda x: x["from"] == "human", conversation))[0]["value"].replace("<audio>", "").replace("\n", "")
    data = pd.read_json(f"{dataset_path}/MULTIS/audio_conversation_10k.json")
    data["question"] = data["conversations"].apply(extract_human)
    data["answer"] = data["conversations"].apply(extract_gpt)
    data["caption"] = data.apply(lambda x: f"{x['question']} {x['answer']}", axis=1)
    data["split"] = "train"
    data["file_name"] = data["video_id"].apply(lambda x: f"{x}.wav")
    data = data[["file_name", "caption", "split"]]
    return data


def AudioDiffCaps():
    adc_rain_dev = pd.read_csv(f"{dataset_path}/AudioDiffCaps/csv/adc_rain_dev.csv")
    adc_rain_dev["split"] = "train"
    adc_rain_eval = pd.read_csv(f"{dataset_path}/AudioDiffCaps/csv/adc_rain_eval.csv")
    adc_rain_eval["split"] = "valid"
    adc_traffic_dev = pd.read_csv(f"{dataset_path}/AudioDiffCaps/csv/adc_traffic_dev.csv")
    adc_traffic_dev["split"] = "train"
    adc_traffic_eval = pd.read_csv(f"{dataset_path}/AudioDiffCaps/csv/adc_traffic_eval.csv")
    adc_traffic_eval["split"] = "valid"
    combined = pd.concat([adc_rain_dev, adc_rain_eval, adc_traffic_dev, adc_traffic_eval], ignore_index=True)
    combined = pd.melt(combined, id_vars=['s1_fn','s2_fn', 'split'], value_vars=['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5'],
            var_name='caption_num', value_name='caption')
    
    combined.rename(columns={"s1_fn": "file_name", "s2_fn": "file_name2"}, inplace=True)

    combined = combined[["file_name", "file_name2", "caption", "split"]]
    
    return combined

def Syncaps():
    df = pd.read_csv(f"{dataset_path}/Syncaps/syncaps_metadata.csv")
    df["split"] = "train"
    df = df[["file_name", "caption", "split"]]
    return df

def BATON():
    df = pd.read_csv(f"{dataset_path}/BATON/baton.csv")
    df["split"] = "train"
    df = df[["file_name", "caption", "split"]]
    return df

def SpatialSoundQA():
    df = pd.read_csv(f"{dataset_path}/SpatialSoundQA/processed_audio.csv")
    df["split"] = df["audio_id"].apply(lambda x: "valid" if x.startswith("eval/") else "train")
    df["caption"] = df["question"] + " " + df["answer"]
    df["file_name"] = df["index"].apply(lambda x: f"{x}.wav")

    df = df[["file_name", "caption", "split"]]
    return df

def ClothoChatGPTMixup():
    df = pd.read_csv(f"{dataset_path}/ClothoChatGPTMixup/mixed_audio_info.csv")
    df["split"] = "train"
    df.rename(columns={"combined_caption": "caption", "filename": "file_name"}, inplace=True)
    df = df[["file_name", "caption", "split"]]
    return df

def LASS():
    # fsd50k_dev_auto_caption.json  fsd50k_eval_auto_caption.json  lass_real_evaluation.csv  lass_synthetic_evaluation.csv  lass_synthetic_validation.csv  lass_validation.json
    # with open(f"{dataset_path}/LASS/fsd50k_dev_auto_caption.json", "r") as file:
    #     dev = json.load(file)
    #     dev = pd.DataFrame(dev["data"])
    # with open(f"{dataset_path}/LASS/fsd50k_eval_auto_caption.json", "r") as file:
    #     eval = json.load(file)
    #     eval = pd.DataFrame(eval["data"])
    
    # dev["split"] = "train"
    # dev.rename(columns={"wav": "file_name"}, inplace=True)
    # dev["file_name"] = dev["file_name"].apply(lambda x: f"FSD50K.dev_audio/{x}")
    # eval["split"] = "test"
    # eval.rename(columns={"wav": "file_name"}, inplace=True)
    # eval["file_name"] = eval["file_name"].apply(lambda x: f"FSD50K.eval_audio/{x}")


    with open(f"{dataset_path}/LASS/lass_validation.json", "r") as file:
        validation = json.load(file)
    validation = pd.DataFrame(validation)
    validation = validation.explode("Captions")
    validation.rename(columns={"Index": "file_name", "Captions": "caption"}, inplace=True)
    validation["split"] = "valid"
    validation["file_name"] = validation["file_name"].apply(lambda x: f"lass_validation/{x}.wav")


    synthetic_val = pd.read_csv(f"{dataset_path}/LASS/lass_synthetic_validation.csv")
    synthetic_val["split"] = "valid"
    synthetic_val.rename(columns={"query": "caption"}, inplace=True)
    synthetic_val["file_name"] = synthetic_val.apply(lambda x: f"synthetic_validation/{x['source']}_{x['noise']}_{x['snr']}.wav", axis=1)


    synthetic_eval = pd.read_csv(f"{dataset_path}/LASS/lass_synthetic_evaluation.csv")
    synthetic_eval["split"] = "test"
    synthetic_eval.rename(columns={"wav": "file_name", "query": "caption"}, inplace=True)
    synthetic_eval["file_name"] = synthetic_eval["file_name"].apply(lambda x: f"lass_evaluation_synth/{x}")
    
    

    real_eval = pd.read_csv(f"{dataset_path}/LASS/lass_real_evaluation.csv")
    real_eval["split"] = "test"
    real_eval.rename(columns={"wav": "file_name", "query": "caption"}, inplace=True)
    real_eval["file_name"] = real_eval["file_name"].apply(lambda x: f"lass_evaluation_real/{x}")


    # combined = pd.concat([dev, eval, real_eval, synthetic_eval, synthetic_val, validation], ignore_index=True)
    combined = pd.concat([real_eval, synthetic_eval, synthetic_val, validation], ignore_index=True)
    combined = combined[["file_name", "caption", "split"]]
    return combined

def AudioCondition():
    # test_strong.json  train_strong.json  val_strong.json
    train = pd.read_json(f"{dataset_path}/AudioCondition/train_strong.json", lines=True)
    val = pd.read_json(f"{dataset_path}/AudioCondition/val_strong.json", lines=True)
    test = pd.read_json(f"{dataset_path}/AudioCondition/test_strong.json", lines=True)

    train["split"] = "train"
    train["file_name"] = train["location"].str.replace("data/strong_audio/", "")
    val["split"] = "valid"
    val["file_name"] = val["location"].str.replace("data/strong_audio/val", "valid")
    test["split"] = "test"
    test["file_name"] = test["location"].str.replace("data/eval/dstrong", "test")

    combined = pd.concat([train, val, test], ignore_index=True)
    combined.rename(columns={"time_captions": "caption"}, inplace=True)
    
    combined = combined[["file_name", "caption", "split"]]
    return combined

def ACalt4():
    df = pd.read_csv(f"{dataset_path}/ACalt4/audiocaps_alternative_4.csv")
    df = pd.melt(df, id_vars=["youtube_id"], value_vars=["caption1", "caption2", "caption3", "caption4"],
            var_name='caption_num', value_name='caption')
    df["file_name"] = "Y" + df["youtube_id"] + ".wav"
    df["split"] = "train"
    df = df[["file_name", "caption", "split"]]
    return df

def PicoAudio():
    # test-frequency-control_onoffFromGpt_multi-event.json   test-onoff-control_multi-event.json   train.json   test-frequency-control_onoffFromGpt_single-event.json  test-onoff-control_single-event.json
    train = pd.read_json(f"{dataset_path}/PicoAudio/train.json", lines=True)
    test_multi = pd.read_json(f"{dataset_path}/PicoAudio/test-onoff-control_multi-event.json", lines=True)
    test_single = pd.read_json(f"{dataset_path}/PicoAudio/test-onoff-control_single-event.json", lines=True)
    test_freq_multi = pd.read_json(f"{dataset_path}/PicoAudio/test-frequency-control_onoffFromGpt_multi-event.json", lines=True)
    test_freq_single = pd.read_json(f"{dataset_path}/PicoAudio/test-frequency-control_onoffFromGpt_single-event.json", lines=True)

    train["split"] = "train"
    test_multi["split"] = "test"
    test_single["split"] = "test"
    test_freq_multi["split"] = "test"
    test_freq_single["split"] = "test"

    combined = pd.concat([train, test_multi, test_single, test_freq_multi, test_freq_single], ignore_index=True)
    combined["file_name"] = combined["filepath"].str.replace("data/", "")
    combined = pd.melt(combined, id_vars=["file_name", "split"], value_vars=["onoffCaption", "frequencyCaption"],
            var_name='caption_type', value_name='caption')
    combined = combined[["file_name", "caption", "split"]]
    return combined

def AudioTime():
    # test500_duration_captions.json  test500_frequency_captions.json  test500_ordering_captions.json  test500_timestamp_captions.json  train5000_duration_captions.json  train5000_frequency_captions.json  train5000_ordering_captions.json  train5000_timestamp_captions.json
    train_duration = pd.read_json(f"{dataset_path}/AudioTime/train5000_duration_captions.json", orient="records")
    def load_json_and_convert(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
            data = [[k, v["caption"], v["event"]] for k,v in data.items()]
            return pd.DataFrame(data, columns=["file_name", "caption", "event"])
    
    train_duration = load_json_and_convert(f"{dataset_path}/AudioTime/train5000_duration_captions.json")
    train_duration["type"] = "train5000_duration"
    train_frequency = load_json_and_convert(f"{dataset_path}/AudioTime/train5000_frequency_captions.json")
    train_frequency["type"] = "train5000_frequency"
    train_ordering = load_json_and_convert(f"{dataset_path}/AudioTime/train5000_ordering_captions.json")
    train_ordering["type"] = "train5000_ordering"
    train_timestamp = load_json_and_convert(f"{dataset_path}/AudioTime/train5000_timestamp_captions.json")
    train_timestamp["type"] = "train5000_timestamp"
    
    test_duration = load_json_and_convert(f"{dataset_path}/AudioTime/test500_duration_captions.json")
    test_duration["type"] = "test500_duration"
    test_frequency = load_json_and_convert(f"{dataset_path}/AudioTime/test500_frequency_captions.json")
    test_frequency["type"] = "test500_frequency"
    test_ordering = load_json_and_convert(f"{dataset_path}/AudioTime/test500_ordering_captions.json")
    test_ordering["type"] = "test500_ordering"
    test_timestamp = load_json_and_convert(f"{dataset_path}/AudioTime/test500_timestamp_captions.json")
    test_timestamp["type"] = "test500_timestamp"

    train_data = pd.concat([train_duration, train_frequency, train_ordering, train_timestamp], ignore_index=True)
    test_data = pd.concat([test_duration, test_frequency, test_ordering, test_timestamp], ignore_index=True)

    train_data["split"] = "train"
    test_data["split"] = "test"

    df = pd.concat([train_data, test_data], ignore_index=True)
    df["file_name"] = df.apply(lambda x: f"{x['type']}/{x['file_name']}.wav", axis=1)
    df = df[["file_name", "caption", "split"]]
    return df

def CompAR():
    df_train = pd.read_json(f"{dataset_path}/CompAR/CompA-R.json")
    df_train["split"] = "train"
    df_train["audio_id"] = df_train["audio_id"].str.replace("./compa_r_train_audios/", "")
    df_train["caption"] = df_train["instruction"] + " " + df_train["output"]
    df_test = pd.read_json(f"{dataset_path}/CompAR/CompA-R-test.json")
    df_test["split"] = "test"
    df_test = df_test.explode("instruction_output")

    df_test["caption"] = df_test["instruction_output"].apply(lambda x: x["instruction"] + " " + x["output"])
    df_train = df_train[["audio_id", "caption", "split"]]
    df_test = df_test[["audio_id", "caption", "split"]]
    df = pd.concat([df_train, df_test], ignore_index=True)
    df["file_name"] = df.apply(lambda x: f"{x['split']}/{x['audio_id']}", axis=1)
    return df

def LAION630k():
    df = pd.read_csv(f"{dataset_path}/LAION630k/combined.csv")
    df["file_name"] = df.apply(lambda x: f"{x['directory']}/{x['subdirectory']}/{x['file_name']}", axis=1)
    df["split"] = "train"
    df.rename(columns={"text": "caption"}, inplace=True)
    df = df[["file_name", "caption", "split"]]
    return df

def AudioAlpaca():
    df = pd.read_csv(f"{dataset_path}/AudioAlpaca/metadata.csv")
    df["split"] = "train"
    df.rename(columns={"filename": "file_name"}, inplace=True)
    df = df[["file_name", "caption", "split"]]
    return df

def Audiostock():
    df = pd.read_csv(f"{dataset_path}/Audiostock/audiostock.csv")
    df.rename(columns={"subdirectory": "split", "text": "caption"}, inplace=True)
    df["file_name"] = df.apply(lambda x: f"{x['split']}/{x['file_name']}", axis=1)
    df = df[["file_name", "caption", "split"]]
    return df

def EpidemicSoundEffects():
    df = pd.read_csv(f"{dataset_path}/EpidemicSoundEffects/epidemic.csv")
    df.rename(columns={"subdirectory": "split", "text": "caption"}, inplace=True)
    df["file_name"] = df.apply(lambda x: f"{x['split']}/{x['file_name']}", axis=1)
    df = df[["file_name", "caption", "split"]]
    return df

def Freesound():
    df = pd.read_csv(f"{dataset_path}/Freesound/freesound.csv")
    df.rename(columns={"subdirectory": "split", "text": "caption"}, inplace=True)
    df["file_name"] = df.apply(lambda x: f"{x['split']}/{x['file_name']}", axis=1)
    df.dropna(inplace=True)
    df = df[["file_name", "caption", "split"]]
    return df

def FreeToUseSounds():
    df = pd.read_csv(f"{dataset_path}/FreeToUseSounds/ftus.csv")
    df.rename(columns={"subdirectory": "split", "text": "caption"}, inplace=True)
    df["file_name"] = df.apply(lambda x: f"{x['split']}/{x['file_name']}", axis=1)
    df = df[["file_name", "caption", "split"]]
    return df

def Paramount():
    df = pd.read_csv(f"{dataset_path}/Paramount/paramount.csv")
    df.rename(columns={"subdirectory": "split", "text": "caption"}, inplace=True)
    df["file_name"] = df.apply(lambda x: f"{x['split']}/{x['file_name']}", axis=1)
    df = df[["file_name", "caption", "split"]]
    return df

def SonnissGameEffects():
    df = pd.read_csv(f"{dataset_path}/SonnissGameEffects/sonniss.csv")
    df.rename(columns={"subdirectory": "split", "text": "caption"}, inplace=True)
    df["file_name"] = df.apply(lambda x: f"{x['split']}/{x['file_name']}", axis=1)
    df = df[["file_name", "caption", "split"]]
    return df

def WeSoundEffects():
    df = pd.read_csv(f"{dataset_path}/WeSoundEffects/we_sound_effects.csv")
    df.rename(columns={"subdirectory": "split", "text": "caption", "id": "file_name"}, inplace=True)
    df["file_name"] = df.apply(lambda x: f"{x['split']}/{x['file_name']}.flac", axis=1)
    df = df[["file_name", "caption", "split"]]
    return df

def BBCSoundEffects():
    df = pd.read_csv(f"{dataset_path}/BBCSoundEffects/bbc.csv")
    df.rename(columns={"subdirectory": "split", "text": "caption", "id": "file_name"}, inplace=True)
    df["file_name"] = df.apply(lambda x: f"{x['split']}/{x['file_name']}", axis=1)
    df = df[["file_name", "caption", "split"]]
    return df


def SoundBible():
    with open(f"{dataset_path}/SoundBible/sb_final.json", "r") as file:
        soundbible = json.load(file)
        soundbible = pd.DataFrame(soundbible["data"])
        soundbible.rename(columns={"id": "file_name"}, inplace=True)
        soundbible.drop(columns=["audio", "download_link", "href", "title", "author", "description"], inplace=True)
    soundbible["split"] = "train"
    soundbible["file_name"] = soundbible["file_name"] + ".flac"
    soundbible = soundbible[["file_name", "caption", "split"]]
    return soundbible

def AudiosetStrong():
    # df = pd.read_csv(f"{dataset_path}/AudiosetStrong/train.csv")
    # df["split"] = "train"
    # df["file_name"] = df["file_name"].apply(lambda x: f"train/{x}")
    # df["file_name"] = df["file_name"].str.replace(".json", ".flac")
    # df.rename(columns={"text": "caption"}, inplace=True)
    # df = df[["file_name", "caption", "split"]]

    # df_test = pd.read_csv(f"{dataset_path}/AudiosetStrong/eval.csv")
    # df_test["split"] = "test"
    # df_test["file_name"] = df_test["file_name"].apply(lambda x: f"test/{x}")
    # df_test["file_name"] = df_test["file_name"].str.replace(".json", ".flac")
    # df_test.rename(columns={"text": "caption"}, inplace=True)
    # df_test = df_test[["file_name", "caption", "split"]]

    # df = pd.concat([df, df_test], ignore_index=True)

    mid_to_display_name = pd.read_csv(f"{dataset_path}/AudiosetStrong/mid_to_display_name.tsv", sep="\t", header=None)
    mid_to_display_name.columns = ["mid", "display_name"]

    df = pd.read_csv(f"{dataset_path}/AudiosetStrong/audioset_train_strong.tsv", sep="\t")
    train_df = df.merge(mid_to_display_name, left_on='label', right_on='mid', how='left')
    train_df.rename(columns={"display_name": "caption"}, inplace=True)
    train_df["file_name"] = train_df["segment_id"].apply(lambda x: f"train/{x}.flac")
    train_df = train_df[["file_name", "caption", "start_time_seconds", "end_time_seconds"]]
    train_df["split"] = "train"

    eval_df = pd.read_csv(f"{dataset_path}/AudiosetStrong/audioset_eval_strong.tsv", sep="\t")
    eval_df = eval_df.merge(mid_to_display_name, left_on='label', right_on='mid', how='left')
    eval_df.rename(columns={"display_name": "caption"}, inplace=True)
    eval_df["file_name"] = eval_df["segment_id"].apply(lambda x: f"test/{x}.flac")
    eval_df = eval_df[["file_name", "caption", "start_time_seconds", "end_time_seconds"]]
    eval_df["split"] = "test"
    
    df = pd.concat([train_df, eval_df], ignore_index=True)
    return df 

def EzAudioCaps():
    df = pd.read_csv(f"{dataset_path}/EzAudioCaps/EzAudioCaps.csv")
    df["split"] = "train"
    df.rename(columns={"audio_path": "file_name"}, inplace=True)

    # Filter filenames that match the pattern audioset_sl_24k/*_[number].wav
    audioset_files = df[df['file_name'].str.contains(r'^audioset_sl_24k/.*_\d+\.wav$', regex=True)]
    
    # Remove _[number].wav to get base filenames
    base_filenames = audioset_files['file_name'].str.replace(r'_\d+\.wav$', '.flac', regex=True) 
    
    # Replace the original filenames with base filenames
    df.loc[audioset_files.index, 'file_name'] = base_filenames

    # Load AudioCaps dataset
    train = pd.read_csv(f"{dataset_path}/AudioCaps/train.csv")
    train["split"] = "train"
    val = pd.read_csv(f"{dataset_path}/AudioCaps/val.csv")
    val["split"] = "valid" 
    test = pd.read_csv(f"{dataset_path}/AudioCaps/test.csv")
    test["split"] = "test"
    audiocaps = pd.concat([train, val, test], ignore_index=True)

    # Create mapping from audiocap_id to youtube_id
    id_mapping = pd.Series(audiocaps.youtube_id.values, index=audiocaps.audiocap_id).to_dict()

    # Get mask for audiocaps subset
    audiocaps_mask = df['file_name'].str.startswith('audiocaps/audiocaps_48k/', na=False)
    
    # Extract IDs from filenames and map to youtube_ids
    df.loc[audiocaps_mask, 'file_name'] = (
        df.loc[audiocaps_mask, 'file_name']
        .str.extract(r'audiocaps/audiocaps_48k/[^/]+/(\d+)\.wav')[0]
        .astype(int)
        .map(id_mapping)
        .apply(lambda x: f"audiocaps/Y{x}.wav")
    )

    # Filter filenames that start with audioset_24k/
    audioset_24k_mask = df['file_name'].str.startswith('audioset_24k/', na=False)
    
    # Remove _number_number.wav pattern from filenames and add Y prefix to filename portion
    df.loc[audioset_24k_mask, 'file_name'] = (
        df.loc[audioset_24k_mask, 'file_name']
        .str.replace(r'_\d+_\d+\.wav$', '.wav', regex=True)
        .str.replace(r'([^/]+)$', r'Y\1', regex=True)
    )

    vggsound_mask = df['file_name'].str.startswith('vggsound_24k/', na=False)
    df.loc[vggsound_mask, 'file_name'] = df.loc[vggsound_mask, 'file_name'].str.replace(".wav", ".flac")

    # Keep only the relevant columns
    df = df[["file_name", "caption", "split"]]

    return df

def AudioHallucination():
    # data/AudioHallucination/Adversarial/data/test-00000-of-00001.parquet
    adversarial_df = pd.read_parquet(f"{dataset_path}/AudioHallucination/Adversarial/data/test-00000-of-00001.parquet")
    # data/AudioHallucination/Popular/data/test-00000-of-00001.parquet
    popular_df = pd.read_parquet(f"{dataset_path}/AudioHallucination/Popular/data/test-00000-of-00001.parquet")
    # data/AudioHallucination/Random/data/test-00000-of-00001.parquet
    random_df = pd.read_parquet(f"{dataset_path}/AudioHallucination/Random/data/test-00000-of-00001.parquet")


    df = pd.concat([adversarial_df, popular_df, random_df], ignore_index=True)
    df["split"] = "test"
    df["file_name"] = df["audio_index"] + ".wav"
    df["caption"] = df["prompt_text"] + " " + df["label"]

    df = df[["file_name", "caption", "split"]]

    return df

def ClothoEntailment():
    # clotho_entailment_development.csv  clotho_entailment_evaluation.csv  clotho_entailment_validation.csv
    dev_df = pd.read_csv(f"{dataset_path}/ClothoEntailment/clotho_entailment_development.csv")
    dev_df["split"] = "train"
    eval_df = pd.read_csv(f"{dataset_path}/ClothoEntailment/clotho_entailment_evaluation.csv")
    eval_df["split"] = "test"
    val_df = pd.read_csv(f"{dataset_path}/ClothoEntailment/clotho_entailment_validation.csv")
    val_df["split"] = "valid"

    df = pd.concat([dev_df, eval_df, val_df], ignore_index=True)
    # Melt the dataframe to create separate rows for each caption type
    df = pd.melt(
        df,
        id_vars=['Audio file', 'split'],
        value_vars=['Entailment', 'Neutral', 'Contradiction'],
        var_name='caption_type',
        value_name='caption_text'
    )

    # Combine caption type and text with prefix
    df['caption'] = df['caption_type'] + ': ' + df['caption_text']

    # Rename audio file column to match schema
    df = df.rename(columns={'Audio file': 'file_name'})
    df["file_name"] = df["split"] + "/" + df["file_name"]
    df = df[["file_name", "caption", "split"]]
    return df


def ClothoMoment():
    # data/ClothoMoment/json/recipe_train.json
    # Load JSON data
    # Load JSON data for each split
    with open(f"{dataset_path}/ClothoMoment/json/recipe_train.json") as f:
        train_data = json.load(f)
    with open(f"{dataset_path}/ClothoMoment/json/recipe_valid.json") as f:
        valid_data = json.load(f)
    with open(f"{dataset_path}/ClothoMoment/json/recipe_test.json") as f:
        test_data = json.load(f)

    # Convert each to dataframe and add split column
    train_df = pd.json_normalize(train_data, sep='_')
    train_df["split"] = "train"
    valid_df = pd.json_normalize(valid_data, sep='_')
    valid_df["split"] = "valid" 
    test_df = pd.json_normalize(test_data, sep='_')
    test_df["split"] = "test"

    # Combine all splits
    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # Drop rows where fg is an empty list
    df = df[df['fg'].map(lambda x: len(x) > 0)]
    
    # Create rows for each foreground item
    df = df.explode('fg')
    
    # Flatten fg dict columns for non-empty fg lists
    fg_df = pd.json_normalize(df['fg'].dropna().tolist(), sep='_')
    
    # Add flattened fg columns back to main df
    df = df.drop('fg', axis=1)
    for col in fg_df.columns:
        df[f'fg_{col}'] = fg_df[col].values
    
    df["fg_end_time"] = df["fg_start_time"] + df["fg_duration"]
    df["caption"] = df["fg_caption"] + " [" + df["fg_start_time"].astype(int).astype(str) + "s, " + df["fg_end_time"].astype(int).astype(str) + "s]"

    df["file_name"] = df["split"] + "/" + df["name"] + ".wav"

    df = df[["file_name", "caption", "split"]]

    return df

def AdobeAuditionSFX():
    df = pd.read_csv(f"{dataset_path}/AdobeAuditionSFX/data.csv")
    df = df[["file_name", "caption", "split"]]
    return df
    
def Zapsplat():
    df = pd.read_csv(f"{dataset_path}/Zapsplat/metadata.csv")
    df["caption"] = df["text"].apply(lambda x: f"{eval(x)[0]}")
    df["split"] = "train"
    df = df[["file_name", "caption", "split"]]
    return df

def ProSoundEffects():
    # data/ProSoundEffects/CORE6-New_Files-All_Tiers.xlsx 
    # data/ProSoundEffects/PSE_CORE6COMP-metadata.xlsx
    df = pd.read_excel(f"{dataset_path}/ProSoundEffects/PSE_CORE6COMP-metadata.xlsx")

    df.rename(columns={"Filename": "file_name", "Description": "caption"}, inplace=True)

    df = df[["file_name", "caption", "split"]]
    return df

def SoundJay():
    df = pd.read_csv(f"{dataset_path}/SoundJay/sound_descriptions.csv")
    df.rename(columns={"filename": "file_name", "description": "caption"}, inplace=True)
    df["split"] = "train"
    df = df[["file_name", "caption", "split"]]
    return df

def RichDetailAudioTextSimulation():
    # Read json as series and convert to dataframe
    series = pd.read_json(f"{dataset_path}/RichDetailAudioTextSimulation/caption_file.json", typ='series')
    df = pd.DataFrame({'file_name': series.index, 'caption': series.values})
    df["file_name"] = df["file_name"] + ".wav"
    df["split"] = "train"
    df = df[["file_name", "caption", "split"]]
    return df

def BigSoundBank():
    df = pd.read_csv(f"{dataset_path}/BigSoundBank/BigSoundBank.csv")
    df["split"] = "train"
    df = df[["file_name", "caption", "split"]]
    return df

def NonSpeech7k():
    df_train = pd.read_csv(f"{dataset_path}/NonSpeech7k/train.csv")
    df_train["split"] = "train"
    df_test = pd.read_csv(f"{dataset_path}/NonSpeech7k/test.csv")
    df_test["split"] = "test"
    df = pd.concat([df_train, df_test], ignore_index=True)
    df["file_name"] = df["split"] + "/" + df["Filename"]
    df.rename(columns={"Filename": "file_name", "Classname": "caption"}, inplace=True)
    df = df[["file_name", "caption", "split"]]
    return df

def FindSounds():
    df = pd.read_csv(f"{dataset_path}/FindSounds/combined_data.csv")
    df["split"] = "train"
    df.rename(columns={"text": "caption"}, inplace=True)
    df = df[["file_name", "caption", "split"]]
    return df

def CHiMEHome():
    df = pd.read_csv(f"{dataset_path}/CHiMEHome/chunk_info.csv")    
    # Read evaluation file to get list of eval files
    eval_files = pd.read_csv(f"{dataset_path}/CHiMEHome/evaluation_chunks_raw.csv", header=None)
    eval_files.columns = ["idx", "file"]
    
    # Remove .48kHz.wav from file_name to match valid file format
    df["split"] = df["file_name"].apply(lambda x: "valid" if x[:-10] in eval_files["file"].tolist() else "train")
    df = df[["file_name", "caption", "split"]]
    return df

def SonycUST():
    df = pd.read_csv(f"{dataset_path}/SonycUST/chunk_info.csv")
    return df

def ESC50():
    df = pd.read_csv(f"{dataset_path}/ESC50/esc50.csv")
    df.rename(columns={"filename": "file_name", "category": "caption"}, inplace=True)
    df["split"] = "train"
    df = df[["file_name", "caption", "split"]]
    return df

if __name__ == "__main__":
    # print(Clotho())
    # print(AudioCaps())
    # print(MACS())
    # print(WavText5K())
    # print(AutoACD())
    # print(AudioCaption())
    # print(SoundDescs())
    # print(WavCaps())
    # print(TextToAudioGrounding())
    # print(FAVDBench())
    # print(ClothoDetail())
    # print(VGGSound())
    # print(SoundingEarth())
    # print(FSD50k())
    # print(Audioset())
    # print(ClothoAQA())
    # print(ClothoV2GPT())
    # print(AudioEgoVLP())
    # print(AFAudioSet())
    # print(SoundVECaps())
    # print(CAPTDURE())
    # print(AnimalSpeak())
    # print(mAQA())
    # print(MULTIS())
    # print(DAQA())
    # print(AudioDiffCaps())
    # print(Syncaps())
    # print(BATON())
    # print(SpatialSoundQA())
    # print(ClothoChatGPTMixup())
    # print(LASS())
    # print(AudioCondition())
    # print(ACalt4())
    # print(PicoAudio())
    # print(AudioTime())
    # print(CompAR())
    # print(LAION630k())
    # print(AudioAlpaca())
    # print(Audiostock())
    # print(EpidemicSoundEffects())
    # print(Freesound())
    # print(FreeToUseSounds())
    # print(Paramount())
    # print(SonnissGameEffects())
    # print(WeSoundEffects())
    # print(BBCSoundEffects())
    # print(SoundBible())
    # print(AudiosetStrong())
    # print(EzAudioCaps())
    # print(AudioHallucination())
    # print(ClothoEntailment())
    # print(ClothoMoment())
    # print(AdobeAuditionSFX())
    # print(Zapsplat())
    # print(ProSoundEffects())
    # print(SoundJay())
    # print(RichDetailAudioTextSimulation())
    # print(BigSoundBank())
    # print(NonSpeech7k())
    # print(FindSounds())
    # print(CHiMEHome())
    print(SonycUST())