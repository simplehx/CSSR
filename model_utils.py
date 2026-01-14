fov2ang = {"0": "left", "90": "front", "180": "right", "270": "back"}
ang2fov = {value: key for key, value in fov2ang.items()}


start_token = "<"
end_token = ">"
pad_token = "?"
geo_code_type = "geohash"
if geo_code_type == "geohash":
    geo_code_char = "0123456789bcdefghjkmnpqrstuvwxyz" + start_token + end_token + pad_token
if geo_code_type == "h3":
    geo_code_char = "0123456789abcdefghijklmnopqrstuvwxyz" + start_token + end_token + pad_token
hash2index = {character: index for index, character in enumerate(geo_code_char)}
index2hash = {index:character for character, index in hash2index.items()}

K_list = [10, 20, 50]

sos_token_id = geo_code_char.find(start_token)
eos_token_id = geo_code_char.find(end_token)
pad_token_id = geo_code_char.find(pad_token)

city_name = "beijing"

train_fovs = [0, 90, 180, 270]
fov2idx = {fov: idx for idx, fov in enumerate(train_fovs)}

model_save_base = "./save/checkpoint/"

TBATCH_SIZE = 512
NUM_WORKERS = 4
EPOCHS = 100
WARMUP_EPOCHS = 5
EMB_DIM = 768
GEO_LENGTH = 6