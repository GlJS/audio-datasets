from urllib.parse import urlparse
import pandas as pd
import yaml
import shutil
import os
import json
from glob import glob
from dotenv import load_dotenv
load_dotenv()

dataset_path = os.getenv("DATASET_PATH")
storage_path = os.getenv("STORAGE_PATH")


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
    train = pd.read_csv(f"{dataset_path}/Auto-ACD/train.csv")
    test = pd.read_csv(f"{dataset_path}/Auto-ACD/test.csv")
    data = pd.concat([train, test], ignore_index=True)
    return data

def AudioCaption():
    hospital = pd.read_json(f"{storage_path}/AudioCaption/hospital_en_all.json")
    hospital['caption_index'] = hospital.groupby('filename').cumcount() + 1
    hospital_zh_dev = pd.read_json(f"{storage_path}/AudioCaption/hospital_zh_dev.json")

    # match on filename and caption_index
    hospital["split"] = hospital.apply(lambda x: "train" if x["filename"] in hospital_zh_dev["filename"].values else "test", axis=1)

    hospital["filename"] = hospital["filename"].str.replace("hospital_3707", "Hospital")
    hospital.rename(columns={"filename": "file_name"}, inplace=True)

    car_dev = pd.read_json(f"{storage_path}/AudioCaption/car_en_dev.json")
    car_dev["split"] = "train"
    car_eval = pd.read_json(f"{storage_path}/AudioCaption/car_en_eval.json")
    car_eval["split"] = "test"
    car = pd.concat([car_dev, car_eval], ignore_index=True)
    car["filename"] = car["filename"].str.replace("car_3610", "Car")
    car.rename(columns={"filename": "file_name"}, inplace=True)
    data = pd.concat([hospital, car], ignore_index=True)
    data = data[["file_name", "caption", "split"]]
    return data

def SoundDescs():
    descriptions = pd.read_pickle(f"{storage_path}/SoundDescs/descriptions.pkl")
    descriptions = pd.DataFrame.from_dict(descriptions, orient="index", columns=["description"])
    categories = pd.read_pickle(f"{storage_path}/SoundDescs/categories.pkl")
    categories = pd.DataFrame.from_dict(categories, orient="index", columns=["category1", "category2", "category3"])
    extra_info = pd.read_pickle(f"{storage_path}/SoundDescs/extra_info.pkl")
    extra_info = pd.DataFrame.from_dict(extra_info, orient="index", columns=["extra_info"])

    train = pd.read_csv(f"{storage_path}/SoundDescs/train_list.txt", header=None, names=["id"])
    train["split"] = "train"
    val = pd.read_csv(f"{storage_path}/SoundDescs/val_list.txt", header=None, names=["id"])
    val["split"] = "valid"
    test = pd.read_csv(f"{storage_path}/SoundDescs/test_list.txt", header=None, names=["id"])
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

# def WavCaps():
#     with open(f"{dataset_path}/WavCaps/as_final.json", "r") as file:
#         audioset = json.load(file)
#         audioset = pd.DataFrame(audioset["data"])
#         audioset.rename(columns={"id": "file_name"}, inplace=True)
#         audioset.drop(columns=["audio"], inplace=True)
#     with open(f"{dataset_path}/WavCaps/bbc_final.json", "r") as file:
#         bbc = json.load(file)
#         bbc = pd.DataFrame(bbc["data"])
#         bbc.rename(columns={"id": "file_name"}, inplace=True)
#         bbc.drop(columns=["description", "category", "audio", "download_link"], inplace=True)
#     with open(f"{dataset_path}/WavCaps/fsd_final.json", "r") as file:
#         fsd = json.load(file)
#         fsd = pd.DataFrame(fsd["data"])
#         fsd.drop(columns=["id", "href", "tags", "author", "description", "audio", "download_link"], inplace=True)
#     with open(f"{dataset_path}/WavCaps/sb_final.json", "r") as file:
#         soundbible = json.load(file)
#         soundbible = pd.DataFrame(soundbible["data"])
#         soundbible.rename(columns={"id": "file_name"}, inplace=True)
#         soundbible.drop(columns=["audio", "download_link"], inplace=True)
#     wavcaps = pd.concat([audioset, bbc, fsd, soundbible], ignore_index=True)
#     return wavcaps

def LAION():
    laion = pd.read_csv(f"{dataset_path}/laion-630k/combined.csv")
    return laion

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
    df = pd.read_csv(f"{dataset_path}/FAVDBench/FAVDBench_Audio.csv")
    # data = glob(f"{storage_path}/FAVDBench/audios/*/*")
    # data_df = pd.DataFrame(data, columns=["file_name"])
    # data_df["audio_id"] = data_df["file_name"].apply(lambda x: os.path.basename(x).split(".")[0])
    # data_df["file_name"] = data_df["file_name"].apply(lambda x: x.replace(f"{storage_path}FAVDBench/audios/", ""))
    # df = pd.merge(df, data_df, left_on="id", right_on="audio_id")
    df.drop(columns=["gpt4"], inplace=True)
    df.rename(columns={"concatenated": "caption", "id": "file_name"}, inplace=True)
    df["split"] = "train"
    # if file_name starts with val, then it is validation
    df.loc[df["file_name"].str.startswith("val"), "split"] = "valid"
    df.loc[df["file_name"].str.startswith("test"), "split"] = "test"
    df["file_name"] = df["file_name"].str.replace("train/", "")
    df["file_name"] = df["file_name"].str.replace("val/", "")
    df["file_name"] = df["file_name"].str.replace("test/", "")
    
    return df

def ClothoDetail():
    with open(f'{storage_path}/ClothoDetail/Clotho-detail-annotation.json', 'r') as file:
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
    df = pd.read_csv(f"{storage_path}/VGGSound/vggsound.csv", header=None, names=["file_name", "start_sec", "caption", "split"])
    # pad up start_sec column to 6 digits with 0
    df["file_name"] = df.apply(lambda x: f"{x['file_name']}_{str(x['start_sec']).zfill(6)}.flac", axis=1)
    df.drop(columns=["start_sec"], inplace=True)
    return df

def SoundingEarth():
    df = pd.read_csv(f"{storage_path}/SoundingEarth/metadata.csv")
    df["caption"] = df.apply(lambda x: f'{x["title"].rstrip(".")}. {x["description"]}', axis=1)
    df.rename(columns={"key": "file_name"}, inplace=True)
    df["split"] = "train"
    df = df[["file_name", "caption", "split"]]
    return df

def FSD50k():
    dev_data = pd.read_csv(f"{storage_path}/FSD50k/FSD50K.ground_truth/dev.csv")
    dev_data["split"] = "train"
    dev_data["file_name"] = dev_data["fname"].apply(lambda x: f"FSD50K.dev_audio/{x}.wav")
    eval_data = pd.read_csv(f"{storage_path}/FSD50k/FSD50K.ground_truth/eval.csv")
    eval_data["split"] = "test"
    eval_data["file_name"] = dev_data["fname"].apply(lambda x: f"FSD50K.eval_audio/{x}.wav")
    combined = pd.concat([dev_data, eval_data], ignore_index=True)
    
    metadata_dev = pd.read_json(f"{storage_path}/FSD50k/FSD50K.metadata/dev_clips_info_FSD50K.json", orient="index")
    metadata_eval = pd.read_json(f"{storage_path}/FSD50k/FSD50K.metadata/eval_clips_info_FSD50K.json", orient="index")

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
    train = pd.read_csv(f"{storage_path}/ClothoAQA/clotho_aqa_train.csv")
    train["split"] = "train"
    val = pd.read_csv(f"{storage_path}/ClothoAQA/clotho_aqa_val.csv")
    val["split"] = "valid"
    test = pd.read_csv(f"{storage_path}/ClothoAQA/clotho_aqa_test.csv")
    test["split"] = "test"
    combined = pd.concat([train, val, test], ignore_index=True)
    # metadata = pd.read_csv(f"{storage_path}/ClothoAQA/clotho_aqa_metadata.csv")
    # combined = pd.merge(combined, metadata, left_on="audio_id", right_on="file_name")
    combined["caption"] = combined.apply(lambda x: f'{x["QuestionText"]} {x["answer"]}', axis=1)
    combined = combined[["file_name", "caption", "split"]]
    return combined

def CAPTDURE():
    single_source_caption = pd.read_csv(f"{storage_path}/CAPTDURE/single_source_caption.csv", sep="\t")
    mixture_source_caption = pd.read_csv(f"{storage_path}/CAPTDURE/mixture_source_caption.csv", sep="\t")

    single_source_caption["split"] = "single_source"
    mixture_source_caption["split"] = "mixture_source"

    combined = pd.concat([single_source_caption, mixture_source_caption], ignore_index=True)
    combined.rename(columns={"wavfile": "file_name", "caption (en)": "caption"}, inplace=True)
    combined = combined[["file_name", "caption", "split"]]
    return combined

def AnimalSpeak():
    data = pd.read_csv(f"{storage_path}/AnimalSpeak/AnimalSpeak_correct.csv")
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
    binary_test_french = pd.read_csv(f"{storage_path}/mAQA/binary_test_french.csv")
    binary_test_french["split"] = "test"
    binary_test_italian = pd.read_csv(f"{storage_path}/mAQA/binary_test_italian.csv")
    binary_test_italian["split"] = "test"
    binary_train_dutch = pd.read_csv(f"{storage_path}/mAQA/binary_train_dutch.csv")
    binary_train_dutch["split"] = "train"
    binary_train_german = pd.read_csv(f"{storage_path}/mAQA/binary_train_german.csv")
    binary_train_german["split"] = "train"
    binary_train_portuguese = pd.read_csv(f"{storage_path}/mAQA/binary_train_portuguese.csv")
    binary_train_portuguese["split"] = "train"
    binary_val_eng = pd.read_csv(f"{storage_path}/mAQA/binary_val_eng.csv")
    binary_val_eng["split"] = "valid"
    binary_val_hindi = pd.read_csv(f"{storage_path}/mAQA/binary_val_hindi.csv")
    binary_val_hindi["split"] = "valid"
    binary_val_spanish = pd.read_csv(f"{storage_path}/mAQA/binary_val_spanish.csv")
    binary_val_spanish["split"] = "valid"
    single_word_train = pd.read_csv(f"{storage_path}/mAQA/single_word_train.csv")
    single_word_train["split"] = "train"
    binary_test_dutch = pd.read_csv(f"{storage_path}/mAQA/binary_test_dutch.csv")
    binary_test_dutch["split"] = "test"
    binary_test_german = pd.read_csv(f"{storage_path}/mAQA/binary_test_german.csv")
    binary_test_german["split"] = "test"
    binary_test_portuguese = pd.read_csv(f"{storage_path}/mAQA/binary_test_portuguese.csv")
    binary_test_portuguese["split"] = "test"
    binary_train_eng = pd.read_csv(f"{storage_path}/mAQA/binary_train_eng.csv")
    binary_train_eng["split"] = "train"
    binary_train_hindi = pd.read_csv(f"{storage_path}/mAQA/binary_train_hindi.csv")
    binary_train_hindi["split"] = "train"
    binary_train_spanish = pd.read_csv(f"{storage_path}/mAQA/binary_train_spanish.csv")
    binary_train_spanish["split"] = "train"
    binary_val_french = pd.read_csv(f"{storage_path}/mAQA/binary_val_french.csv")
    binary_val_french["split"] = "valid"
    binary_val_italian = pd.read_csv(f"{storage_path}/mAQA/binary_val_italian.csv")
    binary_val_italian["split"] = "valid"
    single_word_val = pd.read_csv(f"{storage_path}/mAQA/single_word_val.csv")
    single_word_val["split"] = "valid"
    binary_test_eng = pd.read_csv(f"{storage_path}/mAQA/binary_test_eng.csv")
    binary_test_eng["split"] = "test"
    binary_test_hindi = pd.read_csv(f"{storage_path}/mAQA/binary_test_hindi.csv")
    binary_test_hindi["split"] = "test"
    binary_test_spanish = pd.read_csv(f"{storage_path}/mAQA/binary_test_spanish.csv")
    binary_test_spanish["split"] = "test"
    binary_train_french = pd.read_csv(f"{storage_path}/mAQA/binary_train_french.csv")
    binary_train_french["split"] = "train"
    binary_train_italian = pd.read_csv(f"{storage_path}/mAQA/binary_train_italian.csv")
    binary_train_italian["split"] = "train"
    binary_val_dutch = pd.read_csv(f"{storage_path}/mAQA/binary_val_dutch.csv")
    binary_val_dutch["split"] = "valid"
    binary_val_german = pd.read_csv(f"{storage_path}/mAQA/binary_val_german.csv")
    binary_val_german["split"] = "valid"
    binary_val_portuguese = pd.read_csv(f"{storage_path}/mAQA/binary_val_portuguese.csv")
    binary_val_portuguese["split"] = "valid"
    single_word_test = pd.read_csv(f"{storage_path}/mAQA/single_word_test.csv")
    single_word_test["split"] = "test"

    all_together = pd.concat([binary_test_french, binary_test_italian, binary_train_dutch, binary_train_german, binary_train_portuguese, binary_val_eng, binary_val_hindi, binary_val_spanish, single_word_train, binary_test_dutch, binary_test_german, binary_test_portuguese, binary_train_eng, binary_train_hindi, binary_train_spanish, binary_val_french, binary_val_italian, single_word_val, binary_test_eng, binary_test_hindi, binary_test_spanish, binary_train_french, binary_train_italian, binary_val_dutch, binary_val_german, binary_val_portuguese, single_word_test])

    all_together["caption"] = all_together.apply(lambda x: f"{x['QuestionText']} {x['answer']}", axis=1)
    all_together = all_together[["file_name", "caption", "split"]]
    return all_together


def DAQA():
    with open(f"{dataset_path}/DAQA/DAQA/daqa_train_questions_answers_5.json", "r") as file:
        train = json.load(file)
    with open(f"{dataset_path}/DAQA/DAQA/daqa_val_questions_answers.json", "r") as file:
        val = json.load(file)
    with open(f"{dataset_path}/DAQA/DAQA/daqa_test_questions_answers.json", "r") as file:
        test = json.load(file)
    
    train_df = pd.DataFrame(train["questions"])
    val_df = pd.DataFrame(val["questions"])
    test_df = pd.DataFrame(test["questions"])

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
    data = pd.read_json(f"{storage_path}/MULTIS/audio_conversation_10k.json")
    data["question"] = data["conversations"].apply(extract_human)
    data["answer"] = data["conversations"].apply(extract_gpt)
    data["caption"] = data.apply(lambda x: f"{x['question']} {x['answer']}", axis=1)
    data["split"] = "train"
    data["file_name"] = data["video_id"].apply(lambda x: f"{x}.wav")
    data = data[["file_name", "caption", "split"]]
    return data


def AudioDiffCaps():
    adc_rain_dev = pd.read_csv(f"{storage_path}/AudioDiffCaps/csv/adc_rain_dev.csv")
    adc_rain_dev["split"] = "train"
    adc_rain_eval = pd.read_csv(f"{storage_path}/AudioDiffCaps/csv/adc_rain_eval.csv")
    adc_rain_eval["split"] = "valid"
    adc_traffic_dev = pd.read_csv(f"{storage_path}/AudioDiffCaps/csv/adc_traffic_dev.csv")
    adc_traffic_dev["split"] = "train"
    adc_traffic_eval = pd.read_csv(f"{storage_path}/AudioDiffCaps/csv/adc_traffic_eval.csv")
    adc_traffic_eval["split"] = "valid"
    combined = pd.concat([adc_rain_dev, adc_rain_eval, adc_traffic_dev, adc_traffic_eval], ignore_index=True)
    combined = pd.melt(combined, id_vars=['s1_fn','s2_fn', 'split'], value_vars=['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5'],
            var_name='caption_num', value_name='caption')
    
    combined.rename(columns={"s1_fn": "file_name", "s2_fn": "file_name2"}, inplace=True)

    combined = combined[["file_name", "file_name2", "caption", "split"]]
    
    return combined



if __name__ == "__main__":
    # print(Clotho())
    # print(AudioCaps())
    print(MACS())
    # print(AudioCaption())
    # print(SoundDescs())
    # print(TextToAudioGrounding())
    # print(ClothoDetail())
    # print(VGGSound())
    # print(FAVDBench())
    # print(SoundingEarth())
    # print(ClothoAQA())
    # print(FSD50k())
    # print(Audioset())
    # print(CAPTDURE())
    # print(mAQA())
    # print(MULTIS())
    # print(DAQA())
    # print(AnimalSpeak())
    # print(AudioDiffCaps())
